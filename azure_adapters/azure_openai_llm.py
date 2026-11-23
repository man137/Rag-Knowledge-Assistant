"""Azure OpenAI chat-completions adapter (LLMProvider).

WHAT: `AzureOpenAIProvider` implements the rag.generator.LLMProvider
protocol on top of the official `openai` package's AzureOpenAI client.
`from_settings` wires AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY /
AZURE_OPENAI_API_VERSION / AZURE_OPENAI_DEPLOYMENT from the environment.

DEPLOYMENT NAME vs MODEL NAME (a classic interview trip-up): on Azure OpenAI
you do NOT call a model directly. An administrator creates a DEPLOYMENT
inside the resource and gives that deployment a name (e.g. "gpt-4o-mini-prod");
the client sends that deployment name, and Azure maps it to the underlying
model behind the scenes. The same model can be deployed under several names
with different content filters or quotas. In the openai SDK you pass the
deployment name where the docs say "model".

NOT TESTED AGAINST LIVE SERVICE: this adapter is exercised only against
injected mock clients in tests/test_azure_adapters.py. It has never seen a
real Azure OpenAI endpoint.

MATHS: none.
FAILURE MODE: an expired or wrong API version returns HTTP 404 with a clear
Azure error — update AZURE_OPENAI_API_VERSION to one your resource supports
(see https://learn.microsoft.com/azure/ai-services/openai/reference for the
current list). The SDK is imported lazily so environments without `openai`
raise a clear RuntimeError from the generator factory, not an import crash.
INTERVIEW ONE-LINER: "On Azure OpenAI you call a deployment name, not a
model name — the deployment is what carries quota, filters and versioning."
"""

from __future__ import annotations

import logging
from typing import Callable, Iterator

log = logging.getLogger(__name__)


class AzureOpenAIProvider:
    """LLMProvider over Azure OpenAI chat completions. NOT TESTED AGAINST
    LIVE SERVICE (mocked in tests only)."""

    name = "azure-openai"

    def __init__(self, endpoint: str, api_key: str, api_version: str,
                 deployment: str, temperature: float = 0.2,
                 client_factory: Callable[[], object] | None = None) -> None:
        if not (endpoint and api_key and api_version and deployment):
            raise ValueError(
                "Azure OpenAI needs AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION and "
                "AZURE_OPENAI_DEPLOYMENT.")
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.deployment = deployment   # deployment NAME, not the model id
        self.temperature = temperature
        if client_factory is not None:
            self._client = client_factory()
        else:
            self._client = None

    @classmethod
    def from_settings(cls, settings) -> "AzureOpenAIProvider":
        return cls(endpoint=settings.azure_openai_endpoint,
                   api_key=settings.azure_openai_api_key,
                   api_version=settings.azure_openai_api_version,
                   deployment=settings.azure_openai_deployment,
                   temperature=settings.temperature)

    def _get_client(self):
        if self._client is None:
            try:
                # VERSION-SENSITIVE: openai >= 1.0 exposes AzureOpenAI with
                # azure_endpoint / api_key / api_version kwargs. If a future
                # SDK renames them, fix here first.
                from openai import AzureOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The 'openai' package is not installed; it is required "
                    "only for the Azure providers. Install with "
                    "'pip install -r requirements.txt'."
                ) from exc
            self._client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        return self._client

    def generate(self, messages, *, stream: bool = False,
                 max_tokens: int = 512,
                 temperature: float = 0.2) -> str | Iterator[str]:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.deployment,          # deployment name, see docstring
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except TypeError:
            # VERSION-SENSITIVE fallback: some SDK builds reject unknown
            # kwargs; retry with the minimal stable set.
            response = client.chat.completions.create(
                model=self.deployment, messages=messages, stream=stream)
        if stream:
            def _stream() -> Iterator[str]:
                for chunk in response:
                    if not getattr(chunk, "choices", None):
                        continue
                    piece = getattr(chunk.choices[0].delta, "content", None)
                    if piece:
                        yield piece
            return _stream()
        return str(response.choices[0].message.content)
