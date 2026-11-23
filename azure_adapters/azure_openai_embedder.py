"""Azure OpenAI embeddings adapter (Embedder protocol).

WHAT: `AzureOpenAIEmbedder` implements the rag.embedder.Embedder protocol
against an Azure OpenAI embeddings deployment (e.g. text-embedding-3-small),
returning L2-normalised float32 vectors so the rest of the stack (dot-product
ranking, MMR) works unchanged.
WHY: the Azure AI Search retriever (azure_ai_search_retriever.py) needs
vectors that match what the index stored; pointing BOTH at the same Azure
embedding deployment keeps query and index vectors in the same space.
MATHS: L2 normalisation at the boundary: v_hat = v / |v|, so dot == cosine
everywhere downstream, exactly like the local SentenceTransformerEmbedder.
FAILURE MODE: mixing embedding spaces silently returns nonsense similarity —
never point this embedder at an index built with BGE vectors. The
AZURE_OPENAI_EMBEDDING_DEPLOYMENT must be the deployment name created in the
portal, not the model name.
NOT TESTED AGAINST LIVE SERVICE: exercised only with a mocked client in
tests/test_azure_adapters.py.
INTERVIEW ONE-LINER: "Embeddings are normalised at the adapter boundary, so
switching between my local BGE and Azure's text-embedding deployments is a
config change, not a maths change."
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

log = logging.getLogger(__name__)


class AzureOpenAIEmbedder:
    """Embedder over an Azure OpenAI embeddings deployment. NOT TESTED
    AGAINST LIVE SERVICE (mocked in tests only)."""

    def __init__(self, endpoint: str, api_key: str, api_version: str,
                 deployment: str, dim: int = 1536,
                 client_factory: Callable[[], object] | None = None) -> None:
        if not (endpoint and api_key and api_version and deployment):
            raise ValueError(
                "Azure OpenAI embeddings need AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION and "
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT.")
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.deployment = deployment     # deployment NAME, not model id
        self._dim = int(dim)
        self._client = client_factory() if client_factory else None

    @property
    def name(self) -> str:
        return f"azure-embedder:{self.deployment}"

    @property
    def dim(self) -> int:
        return self._dim

    def _get_client(self):
        if self._client is None:
            try:
                # VERSION-SENSITIVE: openai >= 1.0 AzureOpenAI client.
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

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self._dim), dtype=np.float32)
        client = self._get_client()
        # VERSION-SENSITIVE: openai >= 1.0 embeddings.create(model=...,
        # input=[...]) returns data[i].embedding in input order.
        response = client.embeddings.create(model=self.deployment,
                                            input=list(texts))
        vectors = np.asarray([row.embedding for row in response.data],
                             dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        # Azure text-embedding deployments take no BGE-style instruction
        # prefix; queries and documents share one text form.
        return self._encode(texts)
