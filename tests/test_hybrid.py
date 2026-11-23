"""Tests for rag.hybrid: hand-computed RRF, min-max fusion, modes, ACL."""

from __future__ import annotations

import pytest

from rag.bm25 import BM25Index
from rag.chunking import Chunk
from rag.embedder import FakeEmbedder
from rag.hybrid import MODES, BM25Retriever, DenseRetriever, HybridRetriever, RetrievalResult
from rag.vector_store import VectorStore


def _chunk(cid: str, text: str, roles: list[str]) -> Chunk:
    return Chunk(chunk_id=cid, text=text, source=f"{cid}.md", title=cid,
                 page=1, start_char=0, end_char=len(text),
                 token_count=len(text.split()), allowed_roles=roles)


class _StubRetriever:
    """Returns fixed results, ignoring the query — for fusion arithmetic."""

    def __init__(self, results):
        self.results = results

    def retrieve(self, query, k, allowed_roles):
        return list(self.results)


def _rrf_fixture():
    a = _chunk("A", "alpha text", ["all"])
    b = _chunk("B", "beta text", ["all"])
    # A: rank 1 dense, rank 3 bm25. B: rank 2 in both.
    dense_list = [RetrievalResult(a, 0.9, {"dense": 1}, {"dense": 0.9}),
                  RetrievalResult(b, 0.8, {"dense": 2}, {"dense": 0.8})]
    bm25_list = [RetrievalResult(b, 5.0, {"bm25": 2}, {"bm25": 5.0}),
                 RetrievalResult(a, 9.1, {"bm25": 3}, {"bm25": 9.1})]
    return _StubRetriever(dense_list), _StubRetriever(bm25_list), a, b


def test_rrf_matches_hand_computed_example():
    """Arithmetic written out:
    A = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    B = 1/(60+2) + 1/(60+2) = 2/62          = 0.032258
    A beats B even though the raw scores are incomparable scales."""
    dense, bm25, a, b = _rrf_fixture()
    hy = HybridRetriever(dense=dense, bm25=bm25, mode="hybrid", rrf_k=60)
    out = hy.retrieve("any question", k=2, allowed_roles={"all"})
    expected_a = 1 / 61 + 1 / 63
    expected_b = 1 / 62 + 1 / 62
    assert expected_a > expected_b  # the example must actually discriminate
    scores = {r.chunk.chunk_id: r.score for r in out}
    assert scores["A"] == pytest.approx(expected_a)
    assert scores["B"] == pytest.approx(expected_b)
    assert out[0].chunk.chunk_id == "A"
    # per-retriever ranks survive fusion for debugging
    assert out[0].ranks == {"dense": 1, "bm25": 3}
    assert out[0].scores["dense"] == pytest.approx(0.9)
    assert out[0].scores["bm25"] == pytest.approx(9.1)


def test_minmax_fusion_matches_hand_computed_example():
    """min-max per retriever: dense 0.9/0.8 -> A=1.0, B=0.0; bm25 9.1/5.0 ->
    A=1.0, B=0.0; fused (0.5/0.5): A=1.0, B=0.0."""
    dense, bm25, a, b = _rrf_fixture()
    hy = HybridRetriever(dense=dense, bm25=bm25, mode="hybrid_minmax",
                         weights=(0.5, 0.5))
    out = hy.retrieve("q", k=2, allowed_roles={"all"})
    scores = {r.chunk.chunk_id: r.score for r in out}
    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(0.0)


def test_dense_and_bm25_single_modes_pass_through():
    dense, bm25, a, b = _rrf_fixture()
    out = HybridRetriever(dense=dense, bm25=bm25, mode="dense")\
        .retrieve("q", 5, {"all"})
    assert [r.chunk.chunk_id for r in out] == ["A", "B"]
    assert out[0].score == pytest.approx(0.9)
    out = HybridRetriever(dense=dense, bm25=bm25, mode="bm25")\
        .retrieve("q", 5, {"all"})
    assert [r.chunk.chunk_id for r in out] == ["B", "A"]


def _real_hybrid():
    chunks = [
        _chunk("c0", "Refund rules for event cancellation: full refund 30 or "
                      "more days before the event.", ["all"]),
        _chunk("c1", "Visa processing takes 10 to 15 business days standard.",
               ["all"]),
        _chunk("c2", "Maternity leave is 26 weeks per the hr policy.", ["hr"]),
    ]
    embedder = FakeEmbedder()
    vecs = embedder.encode_documents([c.text for c in chunks])
    store = VectorStore(dim=embedder.dim)
    store.add(chunks, vecs)
    bm25 = BM25Index()
    bm25.build(chunks)
    return (HybridRetriever(DenseRetriever(embedder, store),
                             BM25Retriever(bm25), mode="hybrid",
                             candidate_k=3), chunks)


def test_hybrid_over_real_chunks_and_provenance():
    hy, _ = _real_hybrid()
    out = hy.retrieve("refund rule for event cancellation", k=1,
                      allowed_roles={"all", "guest"})
    assert out[0].chunk.chunk_id == "c0"
    # hybrid implies BOTH retrievers contributed ranks
    assert set(out[0].ranks) == {"dense", "bm25"}


def test_hybrid_acl_blocks_hr_chunk_for_guest():
    hy, chunks = _real_hybrid()
    guest = {"all", "guest"}
    out = hy.retrieve("maternity leave weeks", k=3, allowed_roles=guest)
    # the hr chunk can never appear for a guest, whatever the ranks
    assert out
    assert all(r.chunk.chunk_id != "c2" for r in out)
    hr = {"all", "hr"}
    out = hy.retrieve("maternity leave weeks", k=1, allowed_roles=hr)
    assert out[0].chunk.chunk_id == "c2"


def test_output_never_exceeds_k():
    hy, _ = _real_hybrid()
    assert len(hy.retrieve("refund", k=1, allowed_roles={"all"})) == 1


def test_unknown_mode_rejected():
    dense, bm25, _, _ = _rrf_fixture()
    with pytest.raises(ValueError):
        HybridRetriever(dense=dense, bm25=bm25, mode="magic")
    assert "hybrid_rerank" not in MODES  # pipeline-level mode, not retriever


def test_fusion_deterministic_tiebreak():
    """Symmetric ranks (A: dense 1 / bm25 2; B: dense 2 / bm25 1) give equal
    fused scores; the final order must fall back to chunk_id ascending, so it
    is stable across runs."""
    a = _chunk("z9", "alpha text", ["all"])
    b = _chunk("a1", "beta text", ["all"])
    dense = _StubRetriever([RetrievalResult(a, 0.9, {"dense": 1}, {"dense": 0.9}),
                            RetrievalResult(b, 0.8, {"dense": 2}, {"dense": 0.8})])
    bm25 = _StubRetriever([RetrievalResult(b, 9.0, {"bm25": 1}, {"bm25": 9.0}),
                           RetrievalResult(a, 8.0, {"bm25": 2}, {"bm25": 8.0})])
    hy = HybridRetriever(dense=dense, bm25=bm25, mode="hybrid")
    out = hy.retrieve("q", k=2, allowed_roles={"all"})
    assert out[0].score == pytest.approx(out[1].score)  # 1/61 + 1/62 each
    assert [r.chunk.chunk_id for r in out] == ["a1", "z9"]
