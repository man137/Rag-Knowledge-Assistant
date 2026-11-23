"""Tests for rag.bm25: exact-term ranking, hand-checked score, ACL, IO."""

from __future__ import annotations

import math

import pytest

from rag.bm25 import BM25Index
from rag.chunking import Chunk


def _chunk(cid: str, text: str, roles: list[str]) -> Chunk:
    return Chunk(chunk_id=cid, text=text, source=f"{cid}.md", title=cid,
                 page=1, start_char=0, end_char=len(text),
                 token_count=len(text.split()), allowed_roles=roles)


def test_exact_term_document_ranked_first():
    chunks = [
        _chunk("c0", "Visa applications need a valid passport and photo.", ["all"]),
        _chunk("c1", "The refund window follows event cancellation policy rules.", ["all"]),
        _chunk("c2", "Venue deposits are 25 percent of the booking fee.", ["all"]),
    ]
    idx = BM25Index()
    idx.build(chunks)
    hits = idx.search("refund", 3)
    assert hits[0][0].chunk_id == "c1"
    assert hits[0][1] > 0.0


def test_bm25_score_matches_hand_computed_formula():
    """Two tiny documents, one query term, arithmetic done in the test:
    idf(refund) = ln(1 + (N - df + 0.5)/(df + 0.5)) with N=2, df=1 -> ln(2)
    tf=2, dl=2, avgdl=2.5, k1=1.5, b=0.75
    score = ln(2) * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
          = ln(2) * 2*2.5 / (2 + 1.5*(0.25 + 0.6))
    """
    chunks = [_chunk("c0", "refund refund", ["all"]),
              _chunk("c1", "visa visa visa", ["all"])]
    idx = BM25Index()
    idx.build(chunks)
    N, df, tf, dl, avgdl, k1, b = 2, 1, 2, 2, 2.5, 1.5, 0.75
    idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
    expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    hits = idx.search("refund", 2)
    assert hits[0][0].chunk_id == "c0"
    assert hits[0][1] == pytest.approx(expected)
    # the other document does not contain the term at all
    assert all(c.chunk_id != "c1" for c, _ in hits)


def test_repeated_terms_help_but_saturate():
    text = " ".join(["refund"] * 30)
    other = " ".join(["refund"] * 2)
    idx = BM25Index()
    idx.build([_chunk("many", text, ["all"]), _chunk("few", other, ["all"])])
    hits = idx.search("refund", 2)
    assert hits[0][0].chunk_id == "many"
    # k1=1.5 caps the benefit: 30 occurrences do not score 15x the 2-occurrence doc
    assert hits[0][1] / hits[1][1] < 4.0


def test_stopword_only_or_unknown_query_returns_nothing():
    idx = BM25Index()
    idx.build([_chunk("a", "policy document text", ["all"])])
    assert idx.search("the of", 3) == []
    assert idx.search("qqqqzzzz", 3) == []


def test_acl_ranks_only_permitted_chunks():
    chunks = [_chunk("pub", "refund policy for events", ["all"]),
              _chunk("hr", "maternity leave salary bands", ["hr"])]
    idx = BM25Index()
    idx.build(chunks)
    hits = idx.search("maternity leave", 5, allowed_roles={"all", "guest"})
    assert [c.chunk_id for c, _ in hits] == []
    hits_hr = idx.search("maternity leave", 5, allowed_roles={"all", "hr"})
    assert [c.chunk_id for c, _ in hits_hr] == ["hr"]


def test_k_limit_and_tie_determinism():
    idx = BM25Index()
    idx.build([_chunk("a", "alpha beta", ["all"]),
               _chunk("b", "alpha beta", ["all"]),
               _chunk("c", "gamma delta", ["all"])])
    hits = idx.search("alpha", 1)
    assert len(hits) == 1
    # identical texts tie -> deterministic lowest row index first
    assert hits[0][0].chunk_id == "a"


def test_save_load_roundtrip(tmp_path):
    chunks = [_chunk("c0", "refund rules for event cancellation", ["all"]),
              _chunk("c1", "secret salary bands", ["hr"])]
    idx = BM25Index()
    idx.build(chunks)
    path = tmp_path / "bm25.json"
    idx.save(path)
    loaded = BM25Index.load(path)
    assert loaded.k1 == 1.5 and loaded.b == 0.75
    a = idx.search("refund", 2, {"all", "guest"})
    b = loaded.search("refund", 2, {"all", "guest"})
    assert [(c.chunk_id, s) for c, s in a] == [(c.chunk_id, s) for c, s in b]
    assert loaded.chunks[1].allowed_roles == ["hr"]


def test_empty_index():
    idx = BM25Index()
    idx.build([])
    assert idx.search("anything", 3) == []
