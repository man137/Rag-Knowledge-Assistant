"""Embedders: a Protocol, a real SentenceTransformer embedder and a fake.

WHAT: an `Embedder` Protocol (encode_documents / encode_queries / dim / name),
`SentenceTransformerEmbedder` wrapping the pre-trained BAAI/bge-small-en-v1.5
model (batch encode, L2-normalised, BGE instruction prefix on queries only,
on-disk cache keyed by sha1(model + kind + text)) and `FakeEmbedder`, a
deterministic hashing-trick embedder used by every offline test.
WHY: the embedder is one of the three allowed pre-trained parts of the project
(model, optional reranker, LLM); everything around it is hand-written, so it
sits behind a Protocol the rest of the code depends on, never a concrete class.
MATHS: L2 normalisation turns the dot product into cosine similarity:
cos(q, d) = (q . d) / (|q| |d|) = q_hat . d_hat once both are unit vectors,
so the vector store can rank with a single matrix-vector product and skip the
norm division. FakeEmbedder uses the hashing trick: each content token is
mapped (via sha1 -> seeded NumPy RNG) to a fixed random direction in R^dim; a
text is the normalised sum of its token directions, so two texts sharing many
tokens get a high cosine because their sums share many components.
FAILURE MODE: FakeEmbedder similarity is token-overlap, not semantics — it
cannot rank paraphrases above unrelated-but-wordy text and must never be used
for quality claims; it exists so `pytest -q` runs with no model download.
SentenceTransformerEmbedder raises a clear RuntimeError when the library is
not installed instead of an ImportError deep inside the pipeline.
INTERVIEW ONE-LINER: "I L2-normalise vectors at encode time so retrieval is
one dot product, and tests run against a hashing-trick fake embedder so the
suite is hermetic and offline."
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from rag.cleaning import content_tokens

log = logging.getLogger(__name__)

# BGE family models are trained with an instruction prefix on QUERIES only
# (never on documents); adding it to documents hurts accuracy.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    """Anything that can turn text into fixed-width unit vectors."""

    name: str

    @property
    def dim(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Pre-trained sentence embedding model with an on-disk cache.

    The cache is keyed by sha1(model_name + kind + text) so documents and
    queries never collide and switching the model never reuses stale vectors.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 cache_dir: str | Path | None = None,
                 use_cache: bool = True, batch_size: int = 32) -> None:
        self.model_name = model_name
        self._model = None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = bool(use_cache and self.cache_dir is not None)
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return self.model_name

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install with 'pip install -r requirements.txt' or run "
                    "scripts with --fake to use FakeEmbedder."
                ) from exc
            log.info("Loading embedding model %s (first use may download)", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    # ---- cache ---- #
    def _cache_key(self, text: str, kind: str) -> str:
        digest = hashlib.sha1(
            f"{self.model_name}:{kind}:{text}".encode("utf-8"))
        return digest.hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if not self.use_cache or self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}.npy"

    def _cached(self, key: str) -> np.ndarray | None:
        path = self._cache_path(key)
        if path is None or not path.is_file():
            return None
        try:
            return np.load(path)
        except OSError:  # corrupted cache entry -> recompute
            log.warning("Bad cache entry %s; recomputing", path)
            return None

    def _store(self, key: str, vec: np.ndarray) -> None:
        path = self._cache_path(key)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vec)

    # ---- encoding ---- #
    def _encode_batched(self, texts: Sequence[str], kind: str) -> np.ndarray:
        model = self._load()
        dim = model.get_sentence_embedding_dimension()
        out: list[np.ndarray | None] = [None] * len(texts)
        todo: list[tuple[int, str]] = []  # (output slot, uncached text)
        for i, text in enumerate(texts):
            cached = self._cached(self._cache_key(text, kind))
            if cached is not None:
                out[i] = cached
            else:
                todo.append((i, text))
        if todo:
            # normalize_embeddings=True applies L2 so dot == cosine later.
            vecs = model.encode([t for _, t in todo],
                                batch_size=self.batch_size,
                                normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
            for (slot, text), vec in zip(todo, vecs, strict=True):
                out[slot] = np.asarray(vec, dtype=np.float32)
                self._store(self._cache_key(text, kind),
                            np.asarray(vec, dtype=np.float32))
        return np.vstack([v if v is not None
                          else np.zeros(dim, dtype=np.float32)
                          for v in out]).astype(np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode_batched(texts, kind="doc")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self.model_name.startswith("BAAI/bge"):
            texts = [f"{BGE_QUERY_PREFIX}{t}" for t in texts]
        return self._encode_batched(texts, kind="query")


class FakeEmbedder:
    """Deterministic hashing-trick embedder for offline tests.

    Each content token hashes to a fixed random unit direction; a text is the
    normalised sum of its token directions. No model download, fully
    reproducible, and — crucially — similar texts DO score higher because
    similarity is literal token overlap.

    dim=1024 (not 64): with random projections the noise floor between two
    INDEPENDENT texts is ~1/sqrt(dim), while a contained 4-token query vs a
    ~350-token chunk only reaches |q|/|c| ~ 0.09-0.28 — so the dimension must
    be large enough that token-overlap signal stands above noise. Measured on
    the synthetic corpus: in-scope questions score 0.09-0.28, unrelated
    questions stay under ~0.045; fake-based runs therefore use a dense
    no-answer threshold of 0.07 (see rag.pipeline.fake_tuned).
    """

    def __init__(self, dim: int = 1024, seed: int = 42) -> None:
        self._dim = int(dim)
        self._seed = int(seed)
        self._name = "fake-embedder-v1"
        self._token_vecs: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def _token_vector(self, token: str) -> np.ndarray:
        vec = self._token_vecs.get(token)
        if vec is None:
            token_hash = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng((token_hash ^ self._seed) & 0xFFFFFFFF)
            vec = rng.standard_normal(self._dim)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            self._token_vecs[token] = vec
        return vec

    def _embed(self, text: str) -> np.ndarray:
        toks = content_tokens(text)
        vec = np.zeros(self._dim, dtype=np.float64)
        for tok in toks:
            vec += self._token_vector(tok)
        if not toks:
            # token-free text still needs a deterministic vector
            rng = np.random.default_rng(
                (int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
                 ^ self._seed) & 0xFFFFFFFF)
            vec = rng.standard_normal(self._dim)
        norm = np.linalg.norm(vec)
        return (vec / (norm + 1e-12)).astype(np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._embed(t) for t in texts]) if len(texts) \
            else np.zeros((0, self._dim), dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        # queries and documents share the token space; no prefix for fakes
        return self.encode_documents(texts)


def build_embedder(settings, fake: bool = False) -> Embedder:
    """Factory: FakeEmbedder when --fake, EMBEDDING_MODEL=fake, or when
    sentence-transformers is not installed; otherwise the pre-trained model."""
    if fake or getattr(settings, "embedding_model", None) == "fake":
        return FakeEmbedder(seed=settings.seed)
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        log.warning("sentence-transformers not installed; falling back to FakeEmbedder")
        return FakeEmbedder(seed=settings.seed)
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        cache_dir=settings.cache_dir if settings.embedding_cache else None,
        use_cache=settings.embedding_cache,
    )
