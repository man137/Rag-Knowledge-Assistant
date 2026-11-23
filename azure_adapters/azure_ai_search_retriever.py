"""Azure AI Search retriever behind the rag.hybrid Retriever interface.

WHAT: `AzureAISearchRetriever.retrieve(query, k, allowed_roles)` runs a
hybrid query (BM25 text + vector kNN) against an Azure AI Search index and
returns rag.hybrid.RetrievalResult objects, so the pipeline can swap the
local NumPy store for the managed service without touching any other code.
Role-based access control is enforced with an OData filter over an
`allowed_roles` collection field, mirroring the local ACL-before-ranking
contract.
WHY: this is the "how would you do this with Azure AI Search?" answer made
concrete — same interface, same ACL semantics, managed infrastructure.
MATHS: Azure AI Search fuses its BM25 and vector scores internally (its own
hybrid + optional semantic ranker); this adapter only surfaces the fused
@search.score it returns, which is NOT on the same scale as local cosine or
local RRF — thresholds calibrated for local modes do not transfer.
FAILURE MODE: the query vector must come from the SAME embedding model that
indexed the content; passing a BGE query vector against a text-embedding-3
index returns plausible-looking garbage. The SDK surface changed across
versions (see VERSION-SENSITIVE note below), so the adapter carries a
kwargs fallback.
NOT TESTED AGAINST LIVE SERVICE: exercised only with a mocked SearchClient
in tests/test_azure_adapters.py; the OData filter shape is the documented
search.in(...) pattern from the Azure AI Search OData reference.
INTERVIEW ONE-LINER: "My retriever interface means 'switch to Azure AI
Search' is a new class behind the same three-argument method, with the ACL
as an OData filter instead of a NumPy mask."
"""

from __future__ import annotations

import logging
from typing import Callable

from rag.chunking import Chunk
from rag.hybrid import RetrievalResult

log = logging.getLogger(__name__)

DEFAULT_FIELD_MAP = {
    "chunk_id": "chunk_id",
    "text": "text",
    "source": "source",
    "title": "title",
    "page": "page",
    "vector": "content_vector",
    "roles": "allowed_roles",
}


class AzureAISearchRetriever:
    """Hybrid retriever over an Azure AI Search index. NOT TESTED AGAINST
    LIVE SERVICE (mocked in tests only)."""

    name = "azure-ai-search"

    def __init__(self, endpoint: str, index_name: str, api_key: str,
                 embedder, field_map: dict[str, str] | None = None,
                 client_factory: Callable[[], object] | None = None) -> None:
        if not (endpoint and index_name and api_key):
            raise ValueError(
                "Azure AI Search needs SEARCH_ENDPOINT, SEARCH_INDEX and "
                "SEARCH_KEY.")
        self.endpoint = endpoint
        self.index_name = index_name
        self.api_key = api_key
        self.embedder = embedder          # must MATCH the index's vectors
        self.field_map = dict(DEFAULT_FIELD_MAP)
        if field_map:
            self.field_map.update(field_map)
        self._client = client_factory() if client_factory else None

    def _get_client(self):
        if self._client is None:
            try:
                # VERSION-SENSITIVE: azure-search-documents >= 11.4 exposes
                # SearchClient(endpoint, index_name, credential=AzureKeyCredential).
                from azure.core.credentials import AzureKeyCredential
                from azure.search.documents import SearchClient
            except ImportError as exc:
                raise RuntimeError(
                    "The 'azure-search-documents' package is not installed; "
                    "it is required only for the Azure retriever. Install "
                    "with 'pip install -r requirements.txt'."
                ) from exc
            self._client = SearchClient(
                endpoint=self.endpoint, index_name=self.index_name,
                credential=AzureKeyCredential(self.api_key))
        return self._client

    def _filter_for_roles(self, allowed_roles: set[str] | None) -> str | None:
        """OData ACL filter. None = no filter (the caller's documented
        'no ACL' contract, used only by offline tooling). A guest query
        becomes search.in(allowed_roles, 'all,guest', ',')."""
        if allowed_roles is None:
            return None
        roles = ",".join(sorted(allowed_roles))
        # search.in(field, 'a,b', ',') is Azure AI Search's documented
        # membership filter over collection fields
        return f"search.in({self.field_map['roles']}, '{roles}', ',')"

    def retrieve(self, query: str, k: int,
                 allowed_roles: set[str] | None) -> list[RetrievalResult]:
        client = self._get_client()
        vector = self.embedder.encode_queries([query])[0].tolist()
        vector_field = self.field_map["vector"]
        kwargs = {
            "search_text": query,          # BM25 half of Azure's hybrid
            "filter": self._filter_for_roles(allowed_roles),
            "top": k,
            "search_fields": [self.field_map["text"]],
        }
        try:
            # VERSION-SENSITIVE: >= 11.4 uses VectorizedQuery objects; if a
            # future SDK renames these, the TypeError below retries with the
            # plain `vector=` kwarg that 11.3 and earlier accepted.
            from azure.search.documents.models import VectorizedQuery
            kwargs["vector_queries"] = [
                VectorizedQuery(vector=vector, k_nearest_neighbors=k,
                                fields=vector_field)]
        except ImportError:
            kwargs["vector"] = vector
        try:
            results = client.search(**kwargs)
        except TypeError:
            results = client.search(
                search_text=query, vector=vector,
                filter=kwargs["filter"], top=k,
                vector_fields=vector_field)
        out: list[RetrievalResult] = []
        for rank, document in enumerate(results, start=1):
            score = float(document.get("@search.score", 0.0))
            chunk = Chunk(
                chunk_id=str(document.get(self.field_map["chunk_id"], "")),
                text=str(document.get(self.field_map["text"], "")),
                source=str(document.get(self.field_map["source"], "")),
                title=str(document.get(self.field_map["title"], "")),
                page=int(document.get(self.field_map["page"], 1) or 1),
                start_char=int(document.get("start_char", 0) or 0),
                end_char=int(document.get("end_char", 0) or 0),
                token_count=int(document.get("token_count", 0) or 0),
                allowed_roles=list(document.get(self.field_map["roles"],
                                                ["all"]) or ["all"]),
            )
            out.append(RetrievalResult(
                chunk=chunk, score=score, ranks={"azure": rank},
                scores={"azure": score}))
        return out[:k]
