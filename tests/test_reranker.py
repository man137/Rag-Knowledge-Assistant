"""Tests for rag.reranker: overlap ordering, truncation, flags, errors."""

from __future__ import annotations

import sys

import pytest

from rag.chunking import Chunk
from rag.hybrid import RetrievalResult
from rag.reranker import CrossEncoderReranker, FakeReranker, build_reranker


def _res(cid: str, text: str) -> RetrievalResult:
    chunk = Chunk(chunk_id=cid, text=text, source=f"{cid}.md", title=cid,
                  page=1, start_char=0, end_char=len(text),
                  token_count=len(text.split()), allowed_roles=["all"])
    return RetrievalResult(chunk=chunk, score=0.1, ranks={"dense": 1},
                           scores={"dense": 0.1})


def test_fake_reranker_orders_by_token_overlap():
    q = "refund policy for event cancellation"
    results = [_res("a", "Refund rules for event cancellations and refunds."),
               _res("b", "Visa passport application requirements.")]
    out = FakeReranker().rerank(q, results, top_n=1)
    assert out[0].chunk.chunk_id == "a"
    assert 0.0 < out[0].score <= 1.0
    assert out[0].ranks["reranker"] == 1


def test_fake_reranker_truncates_and_keeps_provenance():
    q = "refund cancellation"
    results = [_res("a", "refund"), _res("b", "cancellation refund"),
               _res("c", "visa")]
    out = FakeReranker().rerank(q, results, top_n=2)
    assert len(out) == 2
    for r in out:
        assert "dense" in r.ranks and "reranker" in r.ranks
        assert "dense" in r.scores and "reranker" in r.scores


def test_fake_reranker_empty_and_ties():
    assert FakeReranker().rerank("q", [], top_n=3) == []
    q = "refund"
    tied = [_res("b", "refund policy"), _res("a", "refund policy")]
    out = FakeReranker().rerank(q, tied, top_n=2)
    assert [r.chunk.chunk_id for r in out] == ["a", "b"]  # chunk_id tiebreak


def test_cross_encoder_needs_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        CrossEncoderReranker().rerank("q", [_res("a", "x")], top_n=1)


def test_build_reranker_flag_logic(settings):
    assert isinstance(build_reranker(settings.replace(rerank_enabled=False),
                                     fake=True), FakeReranker)
    assert build_reranker(settings.replace(rerank_enabled=False),
                          fake=False) is None
    built = build_reranker(settings.replace(rerank_enabled=True), fake=False)
    assert isinstance(built, CrossEncoderReranker)
