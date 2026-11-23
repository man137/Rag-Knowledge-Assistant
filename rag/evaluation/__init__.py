"""rag.evaluation — metrics, the sweep runner and prompt A/B testing.

WHAT: package marker for the evaluation subsystem: metrics.py (pure metric
functions with hand-checkable formulas), run_eval.py (chunk-size x mode
sweep over the eval set) and prompt_ab.py (prompt version comparison).
WHY: evaluation is a first-class component, not an afterthought: every tuning
decision (chunk size, mode, prompt version, threshold) is measured on the
same 50-item eval set instead of argued about.
MATHS: see metrics.py docstrings for recall@k, MRR, nDCG@k, groundedness.
FAILURE MODE: metrics are proxies — groundedness measures overlap with the
RETRIEVED chunks, not truth; scores from --fake runs are harness smoke tests,
never quality evidence.
INTERVIEW ONE-LINER: "I treat prompts and retrieval parameters as code: same
eval set, fixed seeds, results written to CSV and a markdown table."
"""

__version__ = "1.0.0"
