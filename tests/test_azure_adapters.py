"""Azure adapter tests — every client is MOCKED, no SDK, no network.

azure_adapters are NOT TESTED AGAINST LIVE SERVICE; these tests verify the
adapter logic (kwargs, ACL filters, mapping, fallbacks) against injected
fakes only.
"""

from __future__ import annotations

import json
import sys
import types

import httpx
import numpy as np
import pytest

from azure_adapters.azure_ai_search_retriever import AzureAISearchRetriever
from azure_adapters.azure_openai_embedder import AzureOpenAIEmbedder
from azure_adapters.azure_openai_llm import AzureOpenAIProvider
from azure_adapters.graph_sharepoint_loader import GraphSharePointLoader, role_map_from_env
from azure_adapters.teams_webhook import post_teams_webhook
from rag.embedder import FakeEmbedder
from rag.hybrid import RetrievalResult

MSGS = [{"role": "user", "content": "question"}]


# --------------------------- Azure OpenAI LLM --------------------------- #

def _chat_client(content="azure answer", pieces=("Hello ", "world"),
                 reject_temperature=False):
    def create(**kwargs):
        if reject_temperature and "temperature" in kwargs:
            raise TypeError("unexpected keyword 'temperature'")
        if kwargs.get("stream"):
            return iter([types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    delta=types.SimpleNamespace(content=p))]) for p in pieces])
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content))])
    return types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))


def _llm(**kwargs):
    params: dict = {"endpoint": "https://example.openai.azure.com",
                    "api_key": "k", "api_version": "2024-06-01",
                    "deployment": "dep"}
    params.update(kwargs)
    return AzureOpenAIProvider(**params)


def test_azure_llm_non_stream():
    provider = _llm(client_factory=lambda: _chat_client("grounded answer"))
    assert provider.generate(MSGS) == "grounded answer"
    assert provider.name == "azure-openai"
    assert provider.deployment == "dep"


def test_azure_llm_stream_yields_deltas():
    provider = _llm(client_factory=lambda: _chat_client())
    pieces = list(provider.generate(MSGS, stream=True))
    assert "".join(pieces) == "Hello world"


def test_azure_llm_retries_without_temperature_on_typeerror():
    provider = _llm(client_factory=lambda: _chat_client(
        "minimal kwargs answer", reject_temperature=True))
    assert provider.generate(MSGS) == "minimal kwargs answer"


def test_azure_llm_requires_full_configuration():
    with pytest.raises(ValueError, match="AZURE_OPENAI"):
        _llm(endpoint="")


def test_azure_llm_from_settings_maps_fields(settings):
    settings = settings.replace(
        azure_openai_endpoint="https://e.openai.azure.com",
        azure_openai_api_key="k", azure_openai_api_version="2024-06-01",
        azure_openai_deployment="chat-dep")
    provider = AzureOpenAIProvider.from_settings(settings)
    assert provider.deployment == "chat-dep"   # deployment, not model name
    assert provider.api_version == "2024-06-01"


# --------------------------- Azure OpenAI embedder --------------------------- #

def _embed_client(vectors):
    def create(**kwargs):
        assert "input" in kwargs
        return types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=list(v)) for v in vectors])
    return types.SimpleNamespace(embeddings=types.SimpleNamespace(
        create=create))


def _embedder(vectors, **kwargs):
    params: dict = {"endpoint": "https://e.openai.azure.com", "api_key": "k",
                    "api_version": "2024-06-01", "deployment": "embed-dep",
                    "dim": 2}
    params.update(kwargs)
    return AzureOpenAIEmbedder(client_factory=lambda: _embed_client(vectors),
                               **params)


def test_azure_embedder_l2_normalises_rows():
    embedder = _embedder([[3.0, 4.0], [0.0, 5.0]])
    matrix = embedder.encode_documents(["a", "b"])
    np.testing.assert_allclose(matrix[0], [0.6, 0.8], atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    assert matrix.dtype == np.float32
    assert embedder.name == "azure-embedder:embed-dep"
    assert embedder.dim == 2
    assert embedder.encode_documents([]).shape == (0, 2)


def test_azure_embedder_queries_share_text_form():
    embedder = _embedder([[3.0, 4.0]])
    qvec = embedder.encode_queries(["question"])
    np.testing.assert_allclose(qvec[0], [0.6, 0.8], atol=1e-5)


def test_azure_embedder_requires_full_configuration():
    with pytest.raises(ValueError, match="AZURE_OPENAI"):
        _embedder([[1.0, 0.0]], endpoint="")


# --------------------------- Azure AI Search retriever --------------------------- #

_SEARCH_DOCS = [
    {"chunk_id": "c0", "text": "Refunds take 14 business days.",
     "source": "faq.md", "title": "FAQ", "page": 1,
     "allowed_roles": ["all"], "@search.score": 0.87},
    {"chunk_id": "c1", "text": "Venue deposit is 25 percent.",
     "source": "venue.md", "title": "Venue", "page": 2,
     "allowed_roles": ["staff", "manager"], "@search.score": 0.41},
]


class _CapturingSearchClient:
    def __init__(self, docs, fail_on=None):
        self.docs = docs
        self.fail_on = fail_on
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on and self.fail_on in kwargs:
            raise TypeError(f"unexpected keyword {self.fail_on!r}")
        return iter(self.docs)


def _retriever(client):
    return AzureAISearchRetriever(
        endpoint="https://s.search.windows.net", index_name="kb",
        api_key="k", embedder=FakeEmbedder(dim=8),
        client_factory=lambda: client)


def test_search_retriever_builds_role_filter_and_maps_results():
    client = _CapturingSearchClient(_SEARCH_DOCS)
    results = _retriever(client).retrieve("refund rules", k=2,
                                         allowed_roles={"all", "guest"})
    assert client.calls, "client.search must be called"
    kwargs = client.calls[0]
    assert kwargs["filter"] == "search.in(allowed_roles, 'all,guest', ',')"
    assert kwargs["top"] == 2
    assert kwargs["search_text"] == "refund rules"
    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert results[0].chunk.chunk_id == "c0"
    assert results[0].ranks == {"azure": 1}
    assert results[0].score == pytest.approx(0.87)
    assert results[1].chunk.allowed_roles == ["staff", "manager"]


def test_search_retriever_none_roles_means_no_filter():
    client = _CapturingSearchClient(_SEARCH_DOCS)
    _retriever(client).retrieve("q", k=1, allowed_roles=None)
    assert client.calls[0]["filter"] is None


def test_search_retriever_falls_back_to_legacy_vector_kwarg(monkeypatch):
    """Simulate an SDK whose VectorizedQuery kwargs the client rejects: the
    adapter retries with the legacy `vector=` form."""
    fake_models = types.SimpleNamespace(VectorizedQuery=lambda **kw: object())
    stubs = {
        "azure": types.SimpleNamespace(),
        "azure.search": types.SimpleNamespace(),
        "azure.search.documents": types.SimpleNamespace(),
        "azure.search.documents.models": fake_models,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    client = _CapturingSearchClient(_SEARCH_DOCS, fail_on="vector_queries")
    results = _retriever(client).retrieve("refund", k=2,
                                         allowed_roles={"all"})
    assert len(client.calls) == 2          # first failed, second succeeded
    assert "vector_queries" in client.calls[0]
    assert "vector" in client.calls[1]     # legacy fallback kwarg used
    assert [r.chunk.chunk_id for r in results] == ["c0", "c1"]


def test_search_retriever_requires_full_configuration():
    with pytest.raises(ValueError, match="SEARCH"):
        AzureAISearchRetriever(endpoint="", index_name="i", api_key="k",
                               embedder=FakeEmbedder())


# --------------------------- Graph SharePoint loader --------------------------- #

class _FakeResponse:
    def __init__(self, json_data=None, text=""):
        self._json = json_data
        self.text = text
        self.status_code = 200

    def json(self):
        return self._json

    def raise_for_status(self):
        assert self.status_code == 200


class _FakeGraphHttp:
    listing = {"value": [
        {"id": "1", "name": "hr_compensation.md", "file": {},
         "webUrl": "https://example.sharepoint.com/hr_compensation"},
        {"id": "2", "name": "logo.png", "file": {}},
        {"id": "3", "name": "venue_sop.md", "file": {},
         "webUrl": "https://example.sharepoint.com/venue_sop"},
    ]}

    def __init__(self, file_text="Venue deposit is 25 percent of the fee."):
        self.file_text = file_text
        self.urls: list[str] = []
        self.headers: list[dict] = []

    def get(self, url, headers=None):
        self.urls.append(url)
        self.headers.append(headers)
        if url.endswith(":"):
            return _FakeResponse(json_data=self.listing)
        if url.endswith("/content"):
            return _FakeResponse(text=self.file_text)
        return _FakeResponse(json_data=self.listing)


def _loader(http):
    return GraphSharePointLoader(
        tenant_id="t", client_id="c", client_secret="s", site_id="site-id",
        folder_path="policies", role_map={"hr": ["hr"], "venue": ["staff", "manager"]},
        default_roles=["all"], http=http, token_provider=lambda: "fake-token")


def test_graph_loader_lists_and_loads_with_role_mapping():
    http = _FakeGraphHttp()
    loader = _loader(http)
    files = loader.list_files()
    assert len(files) == 3
    assert "/sites/site-id/drive/root:/policies:/children" in http.urls[0]
    assert http.headers[0]["Authorization"] == "Bearer fake-token"
    docs = loader.load()
    assert len(docs) == 2                     # logo.png skipped
    by_name = {d.metadata["source"]: d for d in docs}
    assert by_name["hr_compensation.md"].allowed_roles == ["hr"]
    assert by_name["venue_sop.md"].allowed_roles == ["staff", "manager"]
    assert by_name["hr_compensation.md"].text.startswith("Venue deposit")
    assert by_name["hr_compensation.md"].metadata["web_url"].startswith(
        "https://example.sharepoint.com")


def test_graph_loader_requires_full_configuration():
    with pytest.raises(ValueError, match="GRAPH"):
        GraphSharePointLoader(tenant_id="", client_id="c", client_secret="s",
                              site_id="s", folder_path="f")


def test_role_map_from_env_parsing():
    mapping = role_map_from_env("policy:all,staff;hr:hr;")
    assert mapping == {"policy": ["all", "staff"], "hr": ["hr"]}


# --------------------------- Teams webhook --------------------------- #

def test_teams_webhook_posts_payload_and_parses_response():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read()
        return httpx.Response(200, json={"result": "ok"})

    out = post_teams_webhook("https://outlook.office.com/webhook/abc",
                             "escalation ticket 7", transport=httpx.MockTransport(handler))
    assert out == {"result": "ok"}
    assert captured["url"].endswith("/webhook/abc")
    # compare the parsed payload: JSON spacing differs between httpx versions
    assert json.loads(captured["body"]) == {"text": "escalation ticket 7"}


def test_teams_webhook_accepts_workflows_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read()
        return httpx.Response(202, text="accepted")

    card = {"type": "message", "attachments": [{"type": "AdaptiveCard"}]}
    out = post_teams_webhook("https://prod-xx.logic.azure.com/flows/y",
                             "unused", payload=card,
                             transport=httpx.MockTransport(handler))
    assert b"AdaptiveCard" in captured["body"]
    assert out["status"] == 202


def test_teams_webhook_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="no flow here")

    with pytest.raises(RuntimeError, match="404"):
        post_teams_webhook("https://example/webhook", "text",
                           transport=httpx.MockTransport(handler))


def test_teams_webhook_rejects_empty_url():
    with pytest.raises(ValueError):
        post_teams_webhook("", "text")
