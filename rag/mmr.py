"""Maximal Marginal Relevance (MMR) for result diversification.

WHAT: `mmr_select(query_vec, candidate_vecs, k, lambda_)` returns row indexes
chosen to be simultaneously relevant to the query and non-redundant with what
is already selected.
WHY: policy documents repeat themselves (headers, boilerplate, near-duplicate
clauses); plain top-k retrieval can spend the whole context budget on three
copies of the same sentence. MMR trades a little relevance for diversity.
MATHS: MMR (Carbonell & Goldstein, 1998), iterative greedy selection:
    next = argmax_{d in C\\S} [ lambda * sim(q, d)
                               - (1 - lambda) * max_{s in S} sim(d, s) ]
until k candidates are chosen. sim is cosine on L2-normalised vectors.
The FIRST pick (S empty) is pure relevance; every later pick subtracts the
similarity to the most similar already-selected chunk, so near-duplicates are
penalised. lambda = 1 degenerates to plain relevance ranking; lambda = 0
picks purely diverse (possibly irrelevant) chunks.
FAILURE MODE: with FakeEmbedder, "similarity" is token overlap, so duplicates
must be literal in tests. MMR is not wired into the default pipeline modes
(the corpus is short and curated); it is a tested component the demo can turn
on when a document genuinely repeats itself.
INTERVIEW ONE-LINER: "MMR is relevance minus redundancy — a lambda-weighted
trade-off so the four chunks I send to the LLM cover four different facts
instead of the same fact four times."
"""

from __future__ import annotations

import numpy as np


def _normalise_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def mmr_select(query_vec: np.ndarray, candidate_vecs: np.ndarray, k: int,
               lambda_: float = 0.7) -> list[int]:
    """Greedy MMR selection over candidate rows; returns indexes in pick order.

    query_vec: (d,) search vector; candidate_vecs: (n, d); k <= n.
    """
    cands = np.asarray(candidate_vecs, dtype=np.float64)
    if cands.ndim != 2 or cands.shape[0] == 0:
        return []
    q = np.asarray(query_vec, dtype=np.float64).reshape(-1)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    cands = _normalise_rows(cands)
    n = cands.shape[0]
    k = min(k, n)
    relevance = cands @ q                    # (n,) cosine to the query
    selected: list[int] = []
    chosen_mask = np.zeros(n, dtype=bool)
    for _ in range(k):
        if not selected:
            # S is empty: the redundancy term is 0, this is plain relevance
            best = int(np.argmax(relevance))
        else:
            sims_to_selected = cands @ cands[selected].T   # (n, |S|)
            redundancy = sims_to_selected.max(axis=1)      # (n,)
            mmr = lambda_ * relevance - (1.0 - lambda_) * redundancy
            mmr[chosen_mask] = -np.inf                      # no re-picking
            best = int(np.argmax(mmr))
        if chosen_mask[best]:
            break  # pragma: no cover - defensive
        selected.append(best)
        chosen_mask[best] = True
    return selected
