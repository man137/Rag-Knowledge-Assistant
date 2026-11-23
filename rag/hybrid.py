"""Hybrid retrieval: dense + BM25 fused with RRF (or min-max weighting).

WHAT: the `Retriever` interface (`retrieve(query, k, allowed_roles) ->
list[RetrievalResult]`), DenseRetriever and BM25Retriever adapters, and
HybridRetriever which fuses the two rankings with Reciprocal Rank Fusion
(default) or weighted min-max normalised scores. Modes: "dense", "bm25",
"hybrid" (RRF) and "hybrid_minmax" (comparison).
WHY: dense retrieval paraphrases well but misses exact identifiers; BM25 is
the opposite. Fusing both is the standard hybrid pattern (and what Azure AI
Search does natively); RRF fuses ranks so the two incomparable score scales
never mix.
MATHS: Reciprocal Rank Fusion (Cormack, Clarke, Buettner, SIGIR 2009):
    score(d) = sum over retrievers r of  1 / (k + rank_r(d))
with k = 60 and rank starting at 1. Worked example used in the unit test:
a chunk ranked 1st by dense and 3rd by BM25 scores 1/61 + 1/63 = 0.03227,
while a chunk ranked 2nd by both scores 1/62 + 1/62 = 0.03226 — the first
wins narrowly even though raw scores are incomparable.
min-max fusion normalises each retriever's scores to [0, 1]:
    s'(d) = (s(d) - min) / (max - min)   (0 if max == min)
then takes the weighted sum; it is exposed for comparison, not used as the
default because absolute BM25 scales shift per query.
FAILURE MODE: RRF is rank-only, so it can never say "nothing is relevant" —
every list has a rank 1. The pipeline therefore runs its no-answer check on
the DENSE score of the top fused chunk, never on the RRF score itself.
INTERVIEW ONE-LINER: "Hybrid = dense + BM25 fused by RRF with k=60, the same
formula Azure AI Search's hybrid mode uses, so I can explain what the managed
service is doing behind the scenes."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Protocol

from rag.bm25 import BM25Index
from rag.chunking import Chunk
from rag.vector_store import VectorStore

log = logging.getLogger(__name__)

MODES = ("dense", "bm25", "hybrid", "hybrid_minmax")


@dataclass
class RetrievalResult:
    """One ranked candidate with per-retriever provenance for debugging."""

    chunk: Chunk
    score: float                      # fused (or single-retriever) score
    ranks: dict[str, int] = field(default_factory=dict)   # e.g. {"dense": 1}
    scores: dict[str, float] = field(default_factory=dict)  # raw per-retriever

    def with_score(self, new_score: float, rank_name: str) -> "RetrievalResult":
        clone = replace(self)
        clone.score = new_score
        clone.ranks = dict(self.ranks)
        clone.scores = dict(self.scores)
        clone.ranks[rank_name] = 0  # caller overwrites
        return clone


class Retriever(Protocol):
    """Common retrieval interface used by the pipeline and agent tools."""

    def retrieve(self, query: str, k: int,
                 allowed_roles: set[str] | None) -> list[RetrievalResult]: ...


class DenseRetriever:
    """Cosine search over the NumPy vector store."""

    def __init__(self, embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int,
                 allowed_roles: set[str] | None) -> list[RetrievalResult]:
        vec = self.embedder.encode_queries([query])[0]
        hits = self.store.search(vec, k, allowed_roles)
        results = []
        for rank, (chunk, score) in enumerate(hits, start=1):
            results.append(RetrievalResult(
                chunk=chunk, score=score, ranks={"dense": rank},
                scores={"dense": score}))
        return results


class BM25Retriever:
    """Lexical BM25 search over the same chunks."""

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def retrieve(self, query: str, k: int,
                 allowed_roles: set[str] | None) -> list[RetrievalResult]:
        hits = self.index.search(query, k, allowed_roles)
        results = []
        for rank, (chunk, score) in enumerate(hits, start=1):
            results.append(RetrievalResult(
                chunk=chunk, score=score, ranks={"bm25": rank},
                scores={"bm25": score}))
        return results


class HybridRetriever:
    """Fuse dense and BM25 lists. mode in MODES:
    dense | bm25 | hybrid (RRF) | hybrid_minmax (weighted scores)."""

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever,
                 mode: str = "hybrid", rrf_k: int = 60,
                 weights: tuple[float, float] = (0.5, 0.5),
                 candidate_k: int = 20) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; use {MODES}")
        self.dense = dense
        self.bm25 = bm25
        self.mode = mode
        self.rrf_k = rrf_k
        self.weights = weights
        self.candidate_k = candidate_k

    def retrieve(self, query: str, k: int,
                 allowed_roles: set[str] | None) -> list[RetrievalResult]:
        if self.mode == "dense":
            return self.dense.retrieve(query, k, allowed_roles)
        if self.mode == "bm25":
            return self.bm25.retrieve(query, k, allowed_roles)
        # hybrid modes: fetch a wider candidate pool from each retriever,
        # fuse, then keep the top k
        pool = max(k, self.candidate_k)
        dense_list = self.dense.retrieve(query, pool, allowed_roles)
        bm25_list = self.bm25.retrieve(query, pool, allowed_roles)
        by_chunk: dict[str, RetrievalResult] = {}
        for res in dense_list + bm25_list:
            existing = by_chunk.get(res.chunk.chunk_id)
            if existing is None:
                by_chunk[res.chunk.chunk_id] = res
            else:
                existing.ranks.update(res.ranks)
                existing.scores.update(res.scores)
        fused: list[RetrievalResult] = []
        for res in by_chunk.values():
            if self.mode == "hybrid":
                # RRF: sum 1/(rrf_k + rank) over retrievers that returned it
                score = sum(1.0 / (self.rrf_k + r)
                            for r in res.ranks.values())
            else:  # hybrid_minmax
                score = self._minmax_fused(dense_list, bm25_list, res)
            clone = replace(res)
            clone.score = score
            fused.append(clone)
        # deterministic: score desc, then chunk_id asc
        fused.sort(key=lambda r: (-r.score, r.chunk.chunk_id))
        return fused[:k]

    def _minmax_fused(self, dense_list: list[RetrievalResult],
                      bm25_list: list[RetrievalResult],
                      res: RetrievalResult) -> float:
        """Weighted sum of per-retriever min-max normalised scores."""
        w_dense, w_bm25 = self.weights
        total = 0.0
        for weight, results, name in ((w_dense, dense_list, "dense"),
                                      (w_bm25, bm25_list, "bm25")):
            raw = res.scores.get(name)
            if raw is None or not results:
                continue
            lo = min(r.scores[name] for r in results)
            hi = max(r.scores[name] for r in results)
            norm = 0.0 if hi <= lo else (raw - lo) / (hi - lo)
            total += weight * norm
        return total


def build_retriever(mode: str, embedder, store: VectorStore,
                    bm25: BM25Index, candidate_k: int = 20,
                    rrf_k: int = 60) -> Retriever:
    """Wire the retriever graph for a mode; single-mode objects still come
    back as HybridRetriever instances so callers depend on one interface."""
    return HybridRetriever(
        dense=DenseRetriever(embedder, store),
        bm25=BM25Retriever(bm25),
        mode=mode, rrf_k=rrf_k, candidate_k=candidate_k)
