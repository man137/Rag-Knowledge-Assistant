"""Tests for rag.vector_store: brute force parity, ACL-before-ranking, IO."""

from __future__ import annotations

import numpy as np
import pytest

from rag.chunking import Chunk
from rag.vector_store import VectorStore


def _chunk(cid: str, text: str, roles: list[str]) -> Chunk:
    return Chunk(chunk_id=cid, text=text, source=f"{cid}.md", title=cid,
                 page=1, start_char=0, end_char=len(text),
                 token_count=len(text.split()), allowed_roles=roles)


def _unit_rows(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def test_search_equals_bruteforce_numpy_reference():
    dim, n, k = 16, 60, 5
    chunks = [_chunk(f"c{i:02d}", f"document {i}", ["all"]) for i in range(n)]
    store = VectorStore(dim=dim)
    store.add(chunks, _unit_rows(n, dim, seed=7))
    rng = np.random.default_rng(99)
    for _ in range(10):
        q = rng.standard_normal(dim)
        q = (q / np.linalg.norm(q)).astype(np.float32)
        got = store.search(q, k)
        scores = store.matrix @ q                       # brute-force reference
        ref = np.argsort(-scores, kind="stable")[:k]
        assert [c.chunk_id for c, _ in got] == [chunks[i].chunk_id for i in ref]
        for (_chunk_hit, s), i in zip(got, ref, strict=True):
            assert s == pytest.approx(float(scores[i]), abs=1e-5)


def test_query_is_normalised_defensively():
    chunks = [_chunk("a", "alpha", ["all"]), _chunk("b", "beta", ["all"])]
    store = VectorStore(dim=3)
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    store.add(chunks, vecs)
    got = store.search(np.array([10.0, 0.0, 0.0]), k=1)   # 10x unnormalised
    assert got[0][0].chunk_id == "a"
    assert got[0][1] == pytest.approx(1.0, abs=1e-5)      # cosine, not raw dot


def test_acl_never_returns_restricted_chunk_to_guest():
    store = VectorStore(dim=4)
    public = _chunk("pub", "public policy text", ["all"])
    secret = _chunk("sec", "confidential salary bands", ["hr"])
    q = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)
    q = q / np.linalg.norm(q)
    secret_vec = q.copy()                                # secret MOST relevant
    public_vec = np.array([0.0, 0.0, 0.7, 0.7], dtype=np.float32)
    public_vec /= np.linalg.norm(public_vec)
    store.add([public, secret], np.vstack([public_vec, secret_vec]))
    guest = {"all", "guest"}
    res = store.search(q, k=5, allowed_roles=guest)
    ids = [c.chunk_id for c, _ in res]
    assert ids == ["pub"]                    # only the permitted row survives
    # ranking happens among permitted rows: score is the real cosine, not -inf
    assert res[0][1] == pytest.approx(float(public_vec @ q), abs=1e-5)
    hr = {"all", "hr"}
    assert [c.chunk_id for c, _ in store.search(q, 5, hr)][0] == "sec"


def test_acl_k_larger_than_allowed_returns_exactly_allowed():
    store = VectorStore(dim=4)
    chunks = [_chunk("p1", "a", ["all"]), _chunk("p2", "b", ["all"]),
              _chunk("h1", "c", ["hr"])]
    vecs = _unit_rows(3, 4, seed=1)
    store.add(chunks, vecs)
    res = store.search(vecs[0], k=10, allowed_roles={"all", "guest"})
    assert {c.chunk_id for c, _ in res} == {"p1", "p2"}
    assert len(res) == 2


def test_allowed_roles_none_means_no_filter():
    store = VectorStore(dim=4)
    secret = _chunk("sec", "secret", ["hr"])
    vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    store.add([secret], vec)
    # None is the documented "no ACL" contract for offline tooling only
    assert store.search(np.array([1.0, 0, 0, 0], dtype=np.float32), 1, None)[0][0]\
        .chunk_id == "sec"


def test_zero_allowed_rows_returns_empty():
    store = VectorStore(dim=4)
    store.add([_chunk("h1", "c", ["hr"])], np.ones((1, 4), dtype=np.float32))
    assert store.search(np.ones(4, dtype=np.float32), 3, {"all", "guest"}) == []


def test_delete_removes_rows(tmp_path):
    store = VectorStore(dim=4)
    a1 = _chunk("a1", "x", ["all"])
    a1.source = "a.md"
    a2 = _chunk("a2", "y", ["all"])
    a2.source = "a.md"
    b1 = _chunk("b1", "z", ["all"])
    b1.source = "b.md"
    store.add([a1, a2, b1], np.ones((3, 4), dtype=np.float32))
    assert store.delete("a.md") == 2
    assert [c.chunk_id for c in store.chunks] == ["b1"]
    assert store.matrix.shape == (1, 4)
    assert store.delete("nope.md") == 0


def test_save_load_roundtrip(tmp_path):
    dim = 8
    chunks = [_chunk(f"c{i}", f"text {i}", ["all"] if i % 2 else ["hr"])
              for i in range(5)]
    store = VectorStore(dim=dim)
    store.add(chunks, _unit_rows(5, dim, seed=3))
    path = tmp_path / "store"
    store.save(path)
    loaded = VectorStore.load(path)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in store.chunks]
    np.testing.assert_allclose(loaded.matrix, store.matrix)
    q = _unit_rows(1, dim, seed=4)[0]
    a = store.search(q, 3, {"all", "hr"})
    b = loaded.search(q, 3, {"all", "hr"})
    assert [(c.chunk_id, s) for c, s in a] == [(c.chunk_id, s) for c, s in b]


def test_dimension_mismatch_raises():
    store = VectorStore(dim=4)
    store.add([_chunk("a", "x", ["all"])], np.ones((1, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        store.add([_chunk("b", "y", ["all"])], np.ones((1, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        store.search(np.ones(8, dtype=np.float32), 1)
