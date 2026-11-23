"""Prompt A/B testing: prompt_v1 vs v2 vs v3 on the same eval set.

WHAT: run the same items through the pipeline three times, changing ONLY the
prompt version, then report groundedness, refusal accuracy, injection
resistance and citation validity per version in a markdown table
(prompt_ab.md). Flags mirror run_eval: --fake, --limit, --provider.
WHY: prompts are code; "v3 is better" is a hypothesis until it is measured on
fixed questions with fixed seeds. This script is the evidence for the JD duty
"prompt design and testing".
MATHS: same metric battery as run_eval (means over per-item metrics; see
rag/evaluation/metrics.py). Citation validity uses the valid/max(1,total)
convention, which scores uncited answers 1.0 — v1 asks for no citations at
all, so read its validity together with its (low) citation COUNT.
FAILURE MODE: with --fake the answers are scripted, so only the retrieval
side changes between versions; real comparisons need a real model. Sample
sizes are small (50 items), so treat differences below a few percentage
points as noise and re-run before claiming an ordering.
INTERVIEW ONE-LINER: "I A/B-tested three prompt versions on the same 50
questions and reported groundedness, refusal accuracy, injection resistance
and citation validity — v1/v2/v3 each have a measured trade-off."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

# allow direct execution: python rag/evaluation/prompt_ab.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag.embedder import FakeEmbedder, build_embedder  # noqa: E402
from rag.evaluation.run_eval import doc_roles_map, evaluate_item, load_eval_set  # noqa: E402
from rag.generator import get_provider  # noqa: E402
from rag.hybrid import BM25Retriever, DenseRetriever  # noqa: E402
from rag.loaders import load_directory  # noqa: E402
from rag.pipeline import RAGPipeline, build_index_in_memory, chunk_corpus, fake_tuned  # noqa: E402
from rag.reranker import build_reranker  # noqa: E402

PROMPT_VERSIONS = ("v1", "v2", "v3")
COLUMNS = ["prompt_version", "n_items", "groundedness", "refusal_accuracy",
           "injection_resistance", "citation_validity", "latency_p50_ms"]


def run(settings, fake: bool = False, limit: int | None = None,
        mode: str = "hybrid",
        versions: Sequence[str] = PROMPT_VERSIONS,
        out_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Sweep prompt versions; returns aggregated rows, writes prompt_ab.md."""
    settings = settings.replace(llm_provider="fake") if fake else settings
    if fake:
        settings = fake_tuned(settings)
    settings.apply_seed()
    items = load_eval_set(settings.eval_set_path)
    if limit is not None:
        items = items[:limit]
    docs = load_directory(settings.data_dir)
    roles = doc_roles_map(docs)
    chunks = chunk_corpus(settings, docs)
    embedder = FakeEmbedder(seed=settings.seed) if fake else build_embedder(settings)
    store, bm25 = build_index_in_memory(settings, chunks, embedder)
    reranker = build_reranker(settings)
    llm = get_provider(settings, fake=fake)
    pipeline = RAGPipeline(settings=settings, embedder=embedder,
                          dense=DenseRetriever(embedder, store),
                          bm25=BM25Retriever(bm25), generator=llm,
                          reranker=reranker)
    out_path = Path(out_dir) if out_dir else Path(settings.eval_out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for version in versions:
        rows = []
        for item in items:
            response = pipeline.ask(item["question"], mode=mode, k=4,
                                    user_role=item.get("user_role", "guest"),
                                    prompt_version=version)
            rows.append(evaluate_item(item, response, embedder, roles,
                                      settings.groundedness_threshold))
        agg = {
            "prompt_version": version,
            "n_items": len(rows),
            "groundedness": _mean(rows, "groundedness"),
            "refusal_accuracy": _mean(rows, "refusal_accuracy"),
            "injection_resistance": _mean(rows, "injection_resistance"),
            "citation_validity": _mean(rows, "citation_validity"),
            "latency_p50_ms": _mean(rows, "latency_ms"),
        }
        results.append(agg)
        print(f"prompt_{version}: {agg}")
    write_md(out_path / "prompt_ab.md", results, fake=fake)
    return results


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(mean(values), 4) if values else None


def write_md(path: Path, results: Sequence[dict[str, Any]], fake: bool) -> None:
    lines = [
        "# Prompt A/B results",
        "",
        ("**APPROXIMATE measurements from this run on this corpus; re-run "
         "before quoting.**" if not fake else
         "**HARNESS SMOKE RUN with FakeEmbedder + FakeLLM (--fake): these are "
         "NOT prompt-quality results.**"),
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join(["---"] * len(COLUMNS)) + "|",
    ]
    for row in results:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in COLUMNS) + " |")
    lines += [
        "",
        "Notes: refusal accuracy combines out-of-scope and access-control "
        "items; injection resistance is 1 when the answer contains neither the "
        "canary nor our prompt text; citation validity uses "
        "valid/max(1,total), so UNCITED answers score 1.0 — compare v1 (which "
        "requests no citations) on citation count, not validity alone.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare prompt_v1/v2/v3 on the same eval set.")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from rag.config import Settings

    settings = Settings.from_env()
    if args.provider:
        settings = settings.replace(llm_provider=args.provider)
    run(settings, fake=args.fake, limit=args.limit, mode=args.mode,
        out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
