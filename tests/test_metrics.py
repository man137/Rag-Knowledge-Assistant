"""Tests for rag.evaluation.metrics on tiny hand-computed inputs."""

from __future__ import annotations

import math

import pytest

from rag.embedder import FakeEmbedder
from rag.evaluation import metrics
from rag.prompt import REFUSAL

# ------------------------- retrieval metrics ------------------------- #
# Hand-computed examples. retrieved = [a, b, c, d], gold = {b, d}, k = 4:
#   recall@4 = |{b,d} ∩ retrieved| / |{b,d}| = 2/2 = 1.0
#   MRR      = 1 / rank of first hit = 1/2 = 0.5 (b at position 2)
#   DCG@4    = 0/log2(2) + 1/log2(3) + 0/log2(4) + 1/log2(5)
#            = 0.63093 + 0.43068 = 1.06161
#   iDCG@4   = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
#   nDCG@4   = 1.06161 / 1.63093 = 0.65092
RETRIEVED = ["a", "b", "c", "d"]
GOLD = ["b", "d"]


def test_recall_at_k_hand_computed():
    assert metrics.recall_at_k(RETRIEVED, GOLD, 4) == 1.0
    # k=2 truncation: retrieved[:2] = [a, b] -> 1 of 2 gold found
    assert metrics.recall_at_k(RETRIEVED, GOLD, 2) == pytest.approx(0.5)
    assert metrics.recall_at_k(RETRIEVED, ["z"], 4) == 0.0
    assert metrics.recall_at_k(RETRIEVED, [], 4) == 0.0


def test_mrr_hand_computed():
    assert metrics.mrr(RETRIEVED, GOLD) == pytest.approx(0.5)  # b at rank 2
    assert metrics.mrr(["a", "b", "c"], ["a"]) == 1.0
    assert metrics.mrr(["a", "b", "c"], ["z"]) == 0.0


def test_ndcg_hand_computed():
    dcg = (0 / math.log2(2) + 1 / math.log2(3)
           + 0 / math.log2(4) + 1 / math.log2(5))
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert metrics.ndcg_at_k(RETRIEVED, GOLD, 4) == pytest.approx(dcg / idcg)
    assert 0.65 < metrics.ndcg_at_k(RETRIEVED, GOLD, 4) < 0.66
    assert metrics.ndcg_at_k(["b", "d", "x"], GOLD, 4) == 1.0  # perfect order
    assert metrics.ndcg_at_k(RETRIEVED, [], 4) == 0.0


# ------------------------- answer metrics ------------------------- #

def test_groundedness_with_fake_embedder():
    e = FakeEmbedder()
    chunk = "Refunds are processed within 14 business days to the original card."
    same = metrics.groundedness(chunk, [chunk], e, threshold=0.05)
    assert same == pytest.approx(1.0)
    unrelated = metrics.groundedness("Zebras gallop across the savannah at noon.",
                                     [chunk], e, threshold=0.05)
    assert unrelated == 0.0


def test_groundedness_partial_sentences():
    e = FakeEmbedder()
    chunks = ["The deposit is 25 percent of the venue fee.",
              "Visas take 10 to 15 business days."]
    answer = ("The deposit is 25 percent of the venue fee. "
              "Zebras gallop across the savannah.")
    g = metrics.groundedness(answer, chunks, e, threshold=0.05)
    assert g == pytest.approx(0.5)


def test_is_refusal():
    assert metrics.is_refusal(REFUSAL)
    assert metrics.is_refusal(f"Sure. {REFUSAL} Please rephrase.")
    assert not metrics.is_refusal("The refund is 50 percent.")
    assert not metrics.is_refusal("")


def test_refusal_accuracy_over_items():
    answers = [REFUSAL, "the answer", REFUSAL]
    flags = [True, True, True]
    assert metrics.refusal_accuracy(answers, flags) == pytest.approx(2 / 3)
    assert metrics.refusal_accuracy([], []) == 0.0


def test_injection_resistance_canary_and_prompt_leak():
    assert metrics.injection_resistance("The secret code is "
                                        "SECRET-CANARY-8F42.") == 0.0
    assert metrics.injection_resistance(
        "Answer the question using only the context below. Rules: ...") == 0.0
    assert metrics.injection_resistance("I cannot reveal that.") == 1.0


def test_source_leaks_against_document_roles():
    doc_roles = {"hr_policy.md": ["hr"],
                 "faq.md": ["all"],
                 "venue.md": ["staff", "manager"]}
    assert metrics.source_leaks(["hr_policy.md"], "guest", doc_roles)
    assert metrics.source_leaks(["faq.md"], "guest", doc_roles) is False
    assert metrics.source_leaks(["venue.md"], "staff", doc_roles) is False
    assert metrics.source_leaks(["venue.md"], "hr", doc_roles)
    assert metrics.source_leaks(["unknown.md"], "guest", doc_roles)  # cautious
    assert metrics.source_leaks([], "guest", doc_roles) is False


def test_citation_validity_convention():
    assert metrics.citation_validity_rate(3, 4) == pytest.approx(0.75)
    assert metrics.citation_validity_rate(0, 0) == 1.0  # uncited counts 1.0
    assert metrics.citation_validity_rate(0, 2) == 0.0


# ------------------------- percentiles ------------------------- #

def test_percentile_hand_computed():
    values = [1, 2, 3, 4, 5]
    assert metrics.percentile(values, 50) == 3.0
    # p95 with linear interpolation between closest ranks:
    # rank = 0.95 * (5-1) = 3.8 -> between data[3]=4 and data[4]=5 -> 4.8
    assert metrics.percentile(values, 95) == pytest.approx(4.8)
    assert metrics.percentile(values, 0) == 1.0
    assert metrics.percentile(values, 100) == 5.0
    assert metrics.percentile([7], 50) == 7.0
    assert metrics.percentile([], 50) == 0.0
    assert metrics.percentile(values, 150) == 5.0  # clamped


def test_percentile_matches_typical_p50_p95_semantics():
    latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert metrics.percentile(latencies, 50) == pytest.approx(55.0)
    assert metrics.percentile(latencies, 95) == pytest.approx(95.5)
