"""Tests for rag.mmr: duplicate suppression and the lambda trade-off."""

from __future__ import annotations

import numpy as np

from rag.mmr import mmr_select


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_mmr_skips_near_duplicate_for_diversity():
    rng = np.random.default_rng(5)
    d0 = _unit(rng.standard_normal(8))
    d1 = d0.copy()                                # exact duplicate of d0
    d2 = _unit(rng.standard_normal(8))            # a different direction
    # query sits BETWEEN d0 and d2, so both are relevant...
    q = _unit(d0 + d2)
    cands = np.stack([d0, d1, d2])
    picked = mmr_select(q, cands, k=2, lambda_=0.5)
    # first pick is pure relevance: row 0 wins the tie with its duplicate
    assert picked[0] == 0
    # second pick must NOT be the duplicate: its redundancy against the
    # selected d0 is 1.0, which the lambda penalty outweighs, so the
    # diverse-but-relevant d2 wins instead
    assert picked[1] == 2


def test_mmr_lambda_one_is_plain_relevance():
    rng = np.random.default_rng(6)
    d0 = _unit(rng.standard_normal(8))
    d1 = _unit(d0 + 0.01 * rng.standard_normal(8))  # slightly less aligned
    d2 = _unit(rng.standard_normal(8))
    cands = np.stack([d0, d1, d2])
    q = d0
    assert mmr_select(q, cands, k=3, lambda_=1.0) == [0, 1, 2]


def test_mmr_lambda_zero_picks_maximally_diverse():
    d0 = _unit([1.0, 0.0])
    d1 = _unit([0.0, 1.0])   # orthogonal to d0
    d2 = d0.copy()           # duplicate of the first pick
    cands = np.stack([d0, d1, d2])
    picked = mmr_select(d0, cands, k=2, lambda_=0.0)
    assert picked[0] == 0    # relevance still breaks the first tie
    assert picked[1] == 1    # then pure diversity: the orthogonal row


def test_mmr_k_capped_and_unique():
    rng = np.random.default_rng(7)
    cands = rng.standard_normal((4, 5))
    picked = mmr_select(cands[0], cands, k=10, lambda_=0.7)
    assert sorted(picked) == [0, 1, 2, 3]


def test_mmr_empty_candidates():
    assert mmr_select(np.ones(3), np.zeros((0, 3)), k=5, lambda_=0.5) == []
