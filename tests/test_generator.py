"""Tests for rag.generator: fakes, retries/backoff, streaming, citations."""

from __future__ import annotations

import httpx
import pytest

from rag.generator import (
    FakeLLM,
    OllamaProvider,
    OpenAICompatProvider,
    get_provider,
    parse_citations,
    validate_citations,
)

MSGS = [{"role": "user", "content": "question about refunds"}]


# --------------------------- FakeLLM --------------------------- #

def test_fake_llm_scripted_queue():
    llm = FakeLLM(responses=["first", "second"])
    assert llm.generate(MSGS) == "first"
    assert llm.generate(MSGS) == "second"
    assert len(llm.calls) == 2
    assert llm.calls[0] == MSGS


def test_fake_llm_callable_and_streaming():
    llm = FakeLLM(fn=lambda messages: messages[0]["content"].upper())
    stream = llm.generate(MSGS, stream=True)
    pieces = list(stream)
    assert "".join(pieces) == "QUESTION ABOUT REFUNDS"


def test_fake_llm_default_response():
    llm = FakeLLM()
    assert "[source:" in llm.generate(MSGS)


# --------------------------- citations --------------------------- #

def test_parse_and_validate_citations():
    answer = ("Refunds take 14 business days. [source:0123456789ab] "
              "Also 50 percent. [source:0123456789ab]")
    assert parse_citations(answer) == ["0123456789ab"]   # deduplicated
    valid, invalid = validate_citations(
        answer, ["0123456789ab", "ffffffffffff"])
    assert valid == ["0123456789ab"]
    assert invalid == []


def test_validate_citations_catches_fabricated_id():
    answer = "The deposit is 25 percent [source:deadbeef1234]."
    valid, invalid = validate_citations(answer, ["0123456789ab"])
    assert valid == []
    assert invalid == ["deadbeef1234"]     # the model invented this citation


# --------------------------- Ollama --------------------------- #

def test_ollama_non_stream_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"role": "assistant",
                                                     "content": "the answer"}})
    provider = OllamaProvider("http://ollama.test", "qwen2.5:3b",
                              transport=httpx.MockTransport(handler))
    assert provider.generate(MSGS) == "the answer"
    provider.close()


def test_ollama_retries_with_exponential_backoff():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) < 3:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"message": {"content": "finally"}})

    sleeps: list[float] = []
    provider = OllamaProvider("http://ollama.test", "m", retries=3, backoff=0.5,
                              transport=httpx.MockTransport(handler),
                              sleep=sleeps.append)
    assert provider.generate(MSGS) == "finally"
    assert len(attempts) == 3
    # exponential backoff: attempt 1 waits base*[1,1.5), attempt 2 base*2*[1,1.5)
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] < 0.75
    assert 1.0 <= sleeps[1] < 1.25
    provider.close()


def test_ollama_gives_up_after_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    provider = OllamaProvider("http://ollama.test", "m", retries=2, backoff=0.01,
                              transport=httpx.MockTransport(handler),
                              sleep=lambda d: None)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        provider.generate(MSGS)
    provider.close()


def test_ollama_4xx_is_fatal_not_retried():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(404, json={"error": "model not found"})

    provider = OllamaProvider("http://ollama.test", "m", retries=5, backoff=0.5,
                              transport=httpx.MockTransport(handler),
                              sleep=lambda d: pytest.fail("must not sleep"))
    with pytest.raises(RuntimeError, match="404"):
        provider.generate(MSGS)
    assert len(attempts) == 1
    provider.close()


def test_ollama_5xx_is_retried():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(503, json={"error": "overloaded"})
        return httpx.Response(200, json={"message": {"content": "ok"}})

    provider = OllamaProvider("http://ollama.test", "m", retries=2, backoff=0.01,
                              transport=httpx.MockTransport(handler),
                              sleep=lambda d: None)
    assert provider.generate(MSGS) == "ok"
    assert len(attempts) == 2
    provider.close()


def test_ollama_streaming_yields_tokens():
    body = (b'{"message":{"content":"Refunds "}}\n'
            b'{"message":{"content":"take "}}\n'
            b'{"message":{"content":"14 days."}}\n'
            b'{"done":true}\n')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    provider = OllamaProvider("http://ollama.test", "m",
                              transport=httpx.MockTransport(handler))
    pieces = list(provider.generate(MSGS, stream=True))
    assert "".join(pieces) == "Refunds take 14 days."
    provider.close()


# --------------------------- OpenAI-compatible --------------------------- #

def test_openai_compat_non_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": "compat answer"}}]})

    provider = OpenAICompatProvider("http://compat.test/v1", "m", api_key="test-key",
                                    transport=httpx.MockTransport(handler))
    assert provider.generate(MSGS) == "compat answer"
    provider.close()


def test_openai_compat_stream_parses_sse_and_done_sentinel():
    body = (b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
            b'data: [DONE]\n\n')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    provider = OpenAICompatProvider("http://compat.test/v1", "m",
                                    transport=httpx.MockTransport(handler))
    pieces = list(provider.generate(MSGS, stream=True))
    assert "".join(pieces) == "Hello there"
    provider.close()


# --------------------------- factory --------------------------- #

def test_get_provider_selects_by_settings(settings):
    assert isinstance(get_provider(settings, fake=True), FakeLLM)
    settings = settings.replace(llm_provider="ollama")
    assert isinstance(get_provider(settings), OllamaProvider)
    settings = settings.replace(llm_provider="openai")
    assert isinstance(get_provider(settings), OpenAICompatProvider)


def test_get_provider_azure_lazy_and_helpful(settings, monkeypatch):
    """The Azure provider must fail with an actionable error when the SDK is
    absent, never an unrelated import traceback."""
    import sys

    settings = settings.replace(llm_provider="azure")
    # force the lazy import to fail deterministically (SDK may or may not be
    # installed in a given environment)
    monkeypatch.setitem(sys.modules, "azure_adapters.azure_openai_llm", None)
    with pytest.raises(RuntimeError, match="openai"):
        get_provider(settings)
