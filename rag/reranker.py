"""Optional cross-encoder reranker, behind a settings flag.

WHAT: a `Reranker` protocol over `retrieve`-style candidates: a cross-encoder
(BAAI/bge-reranker-base via sentence-transformers, lazy import) enabled only
when settings.rerank_enabled is true, plus `FakeReranker` (deterministic
token-overlap scoring) used by the offline test-suite. Reranking takes the
top ~20 retrieved candidates down to the final ~4 chunks.
WHY: the cross-encoder is one of the three allowed pre-trained parts of the
project. Bi-encoder retrieval must embed every chunk independently to be
fast; a cross-encoder encodes (query, chunk) JOINTLY, so it sees token-level
interaction and is markedly more accurate — but one forward pass per pair
makes it too slow to run over the whole corpus. Two-stage retrieve-then-
rerank gives both.
MATHS: score(q, d) comes from the model; reranking simply sorts the
candidates by that joint score and keeps the top_n. FakeReranker scores
overlap(content_tokens(q), content_tokens(d)) / |content_tokens(q)| in [0, 1]
— deterministic and monotone in shared terms.
FAILURE MODE: a reranker can only reorder what retrieval already surfaced; a
chunk missed at stage one can never be rescued at stage two. FakeReranker is
not semantic — never use it for quality claims. When sentence-transformers is
absent, CrossEncoderReranker raises a clear RuntimeError instead of failing
deep inside a model call.
INTERVIEW ONE-LINER: "Bi-encoders embed query and document separately for
speed; the cross-encoder scores the pair jointly for accuracy, so you
retrieve fast then rerank the shortlist — the same reason Azure AI Search
added a semantic ranker."
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

from rag.cleaning import content_tokens

log = logging.getLogger(__name__)


class Reranker(Protocol):
    """Reorders retrieval candidates; returns the best top_n."""

    name: str

    def rerank(self, query: str, results: Sequence,
               top_n: int) -> list: ...


class CrossEncoderReranker:
    """BAAI/bge-reranker-base cross-encoder (lazy model load)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return self.model_name

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "sentence-transformers is not installed, so the "
                    "cross-encoder reranker is unavailable. Install with "
                    "'pip install -r requirements.txt' or disable "
                    "RERANK_ENABLED."
                ) from exc
            log.info("Loading reranker %s (first use may download)",
                     self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, results: Sequence, top_n: int) -> list:
        if not results:
            return []
        model = self._load()
        pairs = [(query, r.chunk.text) for r in results]
        scores = model.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(results)),
                       key=lambda i: (-float(scores[i]), results[i].chunk.chunk_id))
        out = []
        for rank, i in enumerate(order[:top_n], start=1):
            clone = _with_rerank(results[i], float(scores[i]), rank)
            out.append(clone)
        return out


class FakeReranker:
    """Deterministic token-overlap reranker for offline tests.

    Score = fraction of the query's content tokens that appear in the chunk.
    """

    def __init__(self) -> None:
        self._name = "fake-reranker-v1"

    @property
    def name(self) -> str:
        return self._name

    def rerank(self, query: str, results: Sequence, top_n: int) -> list:
        if not results:
            return []
        q_tokens = set(content_tokens(query))
        scored = []
        for res in results:
            chunk_tokens = set(content_tokens(res.chunk.text))
            score = (len(q_tokens & chunk_tokens) / len(q_tokens)) if q_tokens else 0.0
            scored.append((score, res))
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk.chunk_id))
        out = []
        for rank, (score, res) in enumerate(scored[:top_n], start=1):
            out.append(_with_rerank(res, score, rank))
        return out


def _with_rerank(result, score: float, rank: int):
    """Copy a RetrievalResult with the reranker score/rank merged in."""
    from dataclasses import replace

    clone = replace(result)
    clone.score = score
    clone.ranks = dict(result.ranks)
    clone.scores = dict(result.scores)
    clone.ranks["reranker"] = rank
    clone.scores["reranker"] = score
    return clone


def build_reranker(settings, fake: bool = False):
    """None unless enabled: fake -> FakeReranker; rerank_enabled -> real model."""
    if fake:
        return FakeReranker()
    if settings.rerank_enabled:
        return CrossEncoderReranker(settings.rerank_model)
    return None
