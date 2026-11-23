"""Tests for the optional IVF index: recall parity and ACL behaviour."""

from __future__ import annotations

import numpy as np

from rag.chunking import Chunk
from rag.vector_store import IVFIndex, VectorStore


def _corpus(n=400, dim=32, seed=11):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    chunks = [Chunk(chunk_id=f"c{i:03d}", text=f"document number {i}",
                    source=f"{i % 6}.md", title="T", page=1, start_char=0,
                    end_char=18, token_count=3,
                    allowed_roles=["all"] if i % 4 else ["hr"])
              for i in range(n)]
    return chunks, v.astype(np.float32)


def _queries(n=10, dim=32, seed=2):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, dim))
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def test_full_probe_recall_is_one():
    chunks, vecs = _corpus()
    store = VectorStore(dim=32)
    store.add(chunks, vecs)
    ivf = IVFIndex(nlist=16, nprobe=16, seed=5)
    ivf.build(chunks, vecs)
    # probing every list scores every row, so top-k must equal brute force
    assert ivf.recall_at_k(_queries(), 10, store) == 1.0


def test_narrow_probe_recall_cannot_exceed_full_probe():
    chunks, vecs = _corpus()
    store = VectorStore(dim=32)
    store.add(chunks, vecs)
    full = IVFIndex(nlist=16, nprobe=16, seed=5)
    full.build(chunks, vecs)
    narrow = IVFIndex(nlist=16, nprobe=1, seed=5)
    narrow.build(chunks, vecs)
    qs = _queries()
    r_full = full.recall_at_k(qs, 10, store)
    r_narrow = narrow.recall_at_k(qs, 10, store)
    assert r_full == 1.0
    assert r_narrow <= r_full


def test_ivf_respects_acl():
    chunks, vecs = _corpus()
    ivf = IVFIndex(nlist=8, nprobe=8, seed=1)
    ivf.build(chunks, vecs)
    guest = {"all", "guest"}
    for q in _queries(5):
        for chunk, _score in ivf.search(q, 10, allowed_roles=guest):
            assert "hr" not in chunk.allowed_roles


def test_build_is_deterministic_for_fixed_seed():
    chunks, vecs = _corpus()
    a = IVFIndex(nlist=8, nprobe=2, seed=9)
    b = IVFIndex(nlist=8, nprobe=2, seed=9)
    a.build(chunks, vecs)
    b.build(chunks, vecs)
    assert a.inverted == b.inverted
    np.testing.assert_allclose(a.centroids, b.centroids)


def test_search_before_build_raises():
    import pytest

    with pytest.raises(RuntimeError):
        IVFIndex().search(np.ones(4), 1)
