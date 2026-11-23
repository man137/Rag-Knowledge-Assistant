"""Tests for rag.embedder (FakeEmbedder determinism + overlap signal)."""

from __future__ import annotations

import numpy as np
import pytest

from rag.embedder import FakeEmbedder


def test_fake_is_deterministic():
    a, b = FakeEmbedder(seed=42), FakeEmbedder(seed=42)
    texts = ["refund policy for events", "visa and passport advisory"]
    np.testing.assert_allclose(a.encode_documents(texts), b.encode_documents(texts))
    np.testing.assert_allclose(a.encode_queries(["question"]), b.encode_queries(["question"]))


def test_fake_shape_and_dtype():
    e = FakeEmbedder(dim=32)
    assert e.dim == 32
    assert e.name == "fake-embedder-v1"
    m = e.encode_documents(["one", "two", "three"])
    assert m.shape == (3, 32)
    assert m.dtype == np.float32
    assert e.encode_queries([]).shape == (0, 32)


def test_fake_prefers_token_overlap():
    # the whole contract: similar texts score higher, so tests can rank
    e = FakeEmbedder()
    q = e.encode_queries(["refund rule for event cancellation"])[0]
    close = e.encode_documents(["refund rules for event cancellations"])[0]
    far = e.encode_documents(["venue deposit insurance checklist"])[0]
    assert float(q @ close) > float(q @ far)


def test_identical_text_scores_one():
    e = FakeEmbedder()
    v1, v2 = e.encode_documents(["identical phrase here", "identical phrase here"])
    assert abs(float(v1 @ v2) - 1.0) < 1e-4


def test_seed_changes_vectors():
    a = FakeEmbedder(seed=1)
    b = FakeEmbedder(seed=2)
    va = a.encode_documents(["alpha beta"])[0]
    vb = b.encode_documents(["alpha beta"])[0]
    assert not np.allclose(va, vb)


@pytest.mark.integration
def test_real_embedder_download_and_cache(tmp_path):
    """Needs network on first run to download BAAI/bge-small-en-v1.5."""
    from rag.embedder import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5",
                                       cache_dir=tmp_path / "cache")
    assert emb.dim == 384
    vecs = emb.encode_documents(["hello world", "refund policy"])
    assert vecs.shape == (2, 384)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)  # L2-normalised -> dot == cosine
