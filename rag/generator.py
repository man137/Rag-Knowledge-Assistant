"""LLM generation: providers, streaming, retries and citation validation.

WHAT: the `LLMProvider` protocol (`generate(messages, stream=...)` returning
either a string or a token generator) with four implementations:
FakeLLM (scriptable, for tests), OllamaProvider (local HTTP, /api/chat),
OpenAICompatProvider (any OpenAI-compatible /chat/completions endpoint) and
AzureOpenAIProvider (lives in azure_adapters/, imported lazily only when
provider=azure). HTTP calls retry with exponential backoff + seeded jitter.
`parse_citations` / `validate_citations` check the model's [source:chunk_id]
citations against the chunk ids actually present in the prompt.
WHY: one protocol means the pipeline, the agent loop and the eval harness can
run identically on the fake, on Ollama on a laptop, or on a hosted endpoint
selected by LLM_PROVIDER env var.
MATHS: retry delay = backoff_base * 2**attempt (+0..backoff_base/2 seeded
jitter), for attempt = 0..retries-1; capped at 30 s per sleep. Exponential
backoff keeps retries polite under transient connection errors and server-side
rate limiting.
FAILURE MODE: a malformed answer can cite chunk ids that were never in the
context (a hallucinated citation is worse than none, because it looks
auditable); validate_citations reports invalid ids so the pipeline can surface
them in the response and in the metrics. Streaming failures mid-answer raise
a RuntimeError after the same retry policy is applied to the whole request.
INTERVIEW ONE-LINER: "All providers sit behind one Protocol with streaming,
timeouts and exponential backoff, and I validate the model's [source:id]
citations against the context actually sent — that is what makes the RAG
auditable instead of just plausible."
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Callable, Iterator, Protocol, Sequence

import httpx

log = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[source:([0-9a-f]{12})\]")


class LLMProvider(Protocol):
    """Chat completion behind one interface, with optional streaming."""

    name: str

    def generate(self, messages: list[dict[str, str]], *, stream: bool = False,
                 max_tokens: int = 512, temperature: float = 0.2,
                 ) -> str | Iterator[str]: ...


class FakeLLM:
    """Scriptable provider for tests and offline demos.

    `responses`: popped in order, one per generate() call.
    `fn`: optional callable messages -> answer (wins when both are given and
    the responses queue is empty). Streaming yields whitespace-split chunks.
    """

    def __init__(self, responses: Sequence[str] | None = None,
                 fn: Callable[[list[dict[str, str]]], str] | None = None) -> None:
        self.responses: list[str] = list(responses or [])
        self.fn = fn
        self.calls: list[list[dict[str, str]]] = []   # every request, for asserts
        self.stream_chunk_size = 3                     # words per yielded token
        self._name = "fake-llm-v1"

    @property
    def name(self) -> str:
        return self._name

    def _next_answer(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if self.responses:
            return self.responses.pop(0)
        if self.fn is not None:
            return self.fn(messages)
        return "Answer from the test fake. [source:000000000000]"

    def generate(self, messages: list[dict[str, str]], *, stream: bool = False,
                 max_tokens: int = 512,
                 temperature: float = 0.2) -> str | Iterator[str]:
        answer = self._next_answer(messages)
        if not stream:
            return answer

        def _tokens() -> Iterator[str]:
            words = answer.split(" ")
            step = self.stream_chunk_size
            for i in range(0, len(words), step):
                piece = " ".join(words[i:i + step])
                if i + step < len(words):
                    piece += " "
                yield piece

        return _tokens()


class _HTTPBase:
    """Shared retry/stream plumbing for the two HTTP providers.

    One httpx.Client per provider instance (thread-safe, connection pooling);
    tests inject a MockTransport. Retryable: transport errors and 5xx.
    Fatal: 4xx (auth/config errors never fix themselves)."""

    def __init__(self, *, timeout: float = 60.0, retries: int = 2,
                 backoff: float = 0.5, seed: int = 42,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] | None = None) -> None:
        self.timeout = timeout
        self.retries = max(0, retries)
        self.backoff = backoff
        self._rng = random.Random(seed)
        self.transport = transport
        self._sleep = sleep if sleep is not None else time.sleep
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            if self.transport is not None:
                self._client = httpx.Client(transport=self.transport,
                                            timeout=self.timeout)
            else:
                self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def post_with_retry(self, url: str, payload: dict, headers: dict,
                        stream: bool) -> httpx.Response:
        """POST with exponential backoff on transport errors and 5xx."""
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            client = self._get_client()
            response: httpx.Response | None = None
            try:
                if stream:
                    request = client.build_request("POST", url, json=payload,
                                                    headers=headers)
                    response = client.send(request, stream=True)
                else:
                    response = client.post(url, json=payload, headers=headers)
                if response.status_code >= 500:
                    last_error = RuntimeError(
                        f"upstream {response.status_code} from {url}")
                    response.close()
                    response = None
                else:
                    response.raise_for_status()
                    return response
            except httpx.HTTPStatusError as exc:
                # 4xx: fatal, do not retry (bad key, bad model, bad request)
                raise RuntimeError(
                    f"LLM HTTP {exc.response.status_code}: {url}"
                ) from exc
            except httpx.HTTPError as exc:      # timeouts, connection errors
                last_error = exc
            if attempt < self.retries:
                delay = min(30.0, self.backoff * (2 ** attempt))
                delay += self._rng.uniform(0, self.backoff / 2.0)
                log.warning("LLM call attempt %d failed (%s); retry in %.2fs",
                            attempt + 1, last_error, delay)
                self._sleep(delay)
        raise RuntimeError(
            f"LLM call failed after {self.retries + 1} attempts: {last_error}")


class OllamaProvider(_HTTPBase):
    """Local Ollama server via POST {OLLAMA_URL}/api/chat.

    Non-streaming returns one JSON body with message.content; streaming
    returns newline-delimited JSON objects, each carrying a token in
    message.content and a done flag.
    """

    def __init__(self, url: str = "http://localhost:11434",
                 model: str = "qwen2.5:3b", **kwargs) -> None:
        super().__init__(**kwargs)
        self.url = url.rstrip("/")
        self.model = model

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def generate(self, messages: list[dict[str, str]], *, stream: bool = False,
                 max_tokens: int = 512,
                 temperature: float = 0.2) -> str | Iterator[str]:
        payload = {
            "model": self.model, "messages": messages, "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        response = self.post_with_retry(f"{self.url}/api/chat", payload,
                                        headers={}, stream=stream)
        if not stream:
            data = json.loads(response.read().decode("utf-8"))
            response.close()
            return str(data.get("message", {}).get("content", ""))

        def _stream() -> Iterator[str]:
            try:
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        yield piece
            finally:
                response.close()

        return _stream()


class OpenAICompatProvider(_HTTPBase):
    """Any OpenAI-compatible endpoint via POST {base_url}/chat/completions
    (vLLM, LiteLLM, OpenRouter, OpenAI itself). Streaming follows the SSE
    convention: lines 'data: {...}' with choices[0].delta.content and the
    'data: [DONE]' sentinel."""

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "gpt-4o-mini", api_key: str = "",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    @property
    def name(self) -> str:
        return f"openai-compat:{self.model}"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(self, messages: list[dict[str, str]], *, stream: bool = False,
                 max_tokens: int = 512,
                 temperature: float = 0.2) -> str | Iterator[str]:
        payload = {"model": self.model, "messages": messages, "stream": stream,
                   "max_tokens": max_tokens, "temperature": temperature}
        response = self.post_with_retry(
            f"{self.base_url}/chat/completions", payload,
            headers=self._headers(), stream=stream)
        if not stream:
            data = json.loads(response.read().decode("utf-8"))
            response.close()
            return str(data["choices"][0]["message"]["content"])

        def _stream() -> Iterator[str]:
            try:
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[len("data:"):].strip()
                    if body == "[DONE]":
                        break
                    if not body:
                        continue
                    delta = json.loads(body)["choices"][0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        yield piece
            finally:
                response.close()

        return _stream()


def parse_citations(answer: str) -> list[str]:
    """All [source:chunk_id] markers in the answer, in order, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for match in CITATION_RE.finditer(answer):
        cid = match.group(1)
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def validate_citations(answer: str,
                       context_chunk_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split cited ids into (valid, invalid) against the ids actually in the
    prompt context. An invalid citation is one the model invented."""
    present = set(context_chunk_ids)
    cited = parse_citations(answer)
    valid = [c for c in cited if c in present]
    invalid = [c for c in cited if c not in present]
    return valid, invalid


def get_provider(settings, fake: bool = False) -> LLMProvider:
    """Factory by settings.llm_provider (or --fake). The Azure provider is
    imported lazily from azure_adapters so this module works without the
    openai SDK installed."""
    provider = settings.llm_provider
    if fake or provider == "fake":
        return FakeLLM()
    if provider == "ollama":
        return OllamaProvider(
            url=settings.ollama_url, model=settings.llm_model,
            timeout=settings.llm_timeout_s, retries=settings.llm_retries,
            backoff=settings.llm_backoff_s, seed=settings.seed)
    if provider == "openai":
        return OpenAICompatProvider(
            base_url=settings.openai_base_url, model=settings.llm_model,
            api_key=settings.openai_api_key, timeout=settings.llm_timeout_s,
            retries=settings.llm_retries, backoff=settings.llm_backoff_s,
            seed=settings.seed)
    if provider == "azure":
        try:
            from azure_adapters.azure_openai_llm import AzureOpenAIProvider
        except ImportError as exc:
            raise RuntimeError(
                "The Azure OpenAI provider requires the 'openai' package "
                "(pip install -r requirements.txt). It is imported lazily "
                "so local runs work without it."
            ) from exc
        return AzureOpenAIProvider.from_settings(settings)
    raise ValueError(f"Unknown llm_provider {provider!r}")
