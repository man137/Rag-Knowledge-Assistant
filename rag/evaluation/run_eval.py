"""The evaluation sweep runner: chunk sizes x retrieval modes.

WHAT: `run_eval.run(settings, ...)` loads data/eval/eval_set.jsonl, chunks
the corpus at each of 256/512/768 tokens (fixed seeds), builds the fake or
real index, asks every eval item in each retrieval mode (dense, bm25, hybrid,
hybrid_rerank), computes the metric battery per item, aggregates per
(chunk_size, mode) and writes results.csv + results.md. CLI flags:
--fake (FakeEmbedder + FakeLLM: smoke-tests the HARNESS, not the system),
--limit N, --provider, --chunk-sizes, --modes, --out.
WHY: this is the artefact that turns "I tuned my RAG" into "I measured it":
same questions, same seeds, one CSV table that shows which combination of
chunk size and retrieval mode actually wins on this corpus.
MATHS: aggregation is the mean of each per-item metric within a type:
recall/MRR/nDCG/groundedness/citation-validity over in_scope items, refusal
accuracy over out_of_scope, injection resistance over injection items, leak
rate over access_control items, and p50/p95 latency over all items (linear
interpolation percentile from rag/evaluation/metrics.py).
FAILURE MODE: numbers from --fake runs are NOT results — FakeEmbedder
retrieval is token overlap and FakeLLM answers are scripted, so the sweep
only proves the harness executes end to end. The student must run WITHOUT
--fake on real hardware to get reportable numbers, and even those are
approximate single-machine measurements.
INTERVIEW ONE-LINER: "I swept chunk size against retrieval mode on a fixed
50-item eval set and wrote the results to CSV and markdown — the numbers in
my resume come from that table."
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

# allow `python rag/evaluation/run_eval.py` from the repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag.embedder import FakeEmbedder, build_embedder  # noqa: E402
from rag.evaluation import metrics  # noqa: E402
from rag.generator import get_provider  # noqa: E402
from rag.hybrid import BM25Retriever, DenseRetriever  # noqa: E402
from rag.loaders import load_directory  # noqa: E402
from rag.pipeline import RAGPipeline, build_index_in_memory, chunk_corpus, fake_tuned  # noqa: E402
from rag.reranker import FakeReranker, build_reranker  # noqa: E402

DEFAULT_CHUNK_SIZES = (256, 512, 768)
DEFAULT_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
EVAL_TYPES = ("in_scope", "out_of_scope", "injection", "access_control")

CSV_COLUMNS = [
    "chunk_size", "mode", "n_items",
    "recall@4", "mrr", "ndcg@4", "groundedness", "citation_validity",
    "refusal_accuracy", "injection_resistance", "leak_rate",
    "latency_p50_ms", "latency_p95_ms", "ttft_p50_ms",
]


def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def doc_roles_map(docs) -> dict[str, list[str]]:
    """file name -> allowed_roles from front matter, for leak checks."""
    return {Path(d.source).name: list(d.allowed_roles) for d in docs}


def evaluate_item(item: dict[str, Any], response, embedder,
                  doc_roles: dict[str, list[str]],
                  groundedness_threshold: float) -> dict[str, float | None]:
    """Per-item metric battery. Shared by run_eval and prompt_ab."""
    retrieved_sources = [Path(s.source).name for s in response.sources]
    total_cites = len(response.valid_citations) + len(response.invalid_citations)
    row: dict[str, float | None] = {
        "recall@4": None, "mrr": None, "ndcg@4": None, "groundedness": None,
        "citation_validity": None, "refusal_accuracy": None,
        "injection_resistance": None, "leak_rate": None,
        "latency_ms": sum(response.timings.values()),
        "ttft_ms": response.ttft_ms,
    }
    if item["type"] == "in_scope":
        gold = item["gold_sources"]
        row["recall@4"] = metrics.recall_at_k(retrieved_sources, gold, 4)
        row["mrr"] = metrics.mrr(retrieved_sources, gold)
        row["ndcg@4"] = metrics.ndcg_at_k(retrieved_sources, gold, 4)
        row["groundedness"] = metrics.groundedness(
            response.answer, [s.text for s in response.sources], embedder,
            threshold=groundedness_threshold)
        row["citation_validity"] = metrics.citation_validity_rate(
            len(response.valid_citations), total_cites)
        # an in-scope item must NOT be refused
        row["refusal_accuracy"] = 0.0 if metrics.is_refusal(response.answer) else 1.0
    elif item["type"] == "out_of_scope":
        row["refusal_accuracy"] = float(
            metrics.is_refusal(response.answer) or response.no_answer)
    elif item["type"] == "injection":
        row["injection_resistance"] = metrics.injection_resistance(
            response.answer)
        row["refusal_accuracy"] = float(metrics.is_refusal(response.answer))
    elif item["type"] == "access_control":
        row["refusal_accuracy"] = float(metrics.is_refusal(response.answer))
        row["leak_rate"] = float(metrics.source_leaks(
            [s.source for s in response.sources], item["user_role"], doc_roles))
    return row


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean of each metric (ignoring None) plus latency percentiles."""

    def _mean(key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(mean(values), 4) if values else None

    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    ttfts = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
    return {
        "n_items": len(rows),
        "recall@4": _mean("recall@4"),
        "mrr": _mean("mrr"),
        "ndcg@4": _mean("ndcg@4"),
        "groundedness": _mean("groundedness"),
        "citation_validity": _mean("citation_validity"),
        "refusal_accuracy": _mean("refusal_accuracy"),
        "injection_resistance": _mean("injection_resistance"),
        "leak_rate": _mean("leak_rate"),
        "latency_p50_ms": round(metrics.percentile(latencies, 50), 3),
        "latency_p95_ms": round(metrics.percentile(latencies, 95), 3),
        "ttft_p50_ms": round(metrics.percentile(ttfts, 50), 3),
    }


def run(settings, fake: bool = False, limit: int | None = None,
        chunk_sizes: Sequence[int] = DEFAULT_CHUNK_SIZES,
        modes: Sequence[str] = DEFAULT_MODES,
        out_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Run the full sweep; returns the aggregated rows and writes CSV + MD."""
    settings = settings.replace(llm_provider="fake") if fake else settings
    if fake:
        settings = fake_tuned(settings)
    settings.apply_seed()
    items = load_eval_set(settings.eval_set_path)
    if limit is not None:
        items = items[:limit]
    docs = load_directory(settings.data_dir)
    roles = doc_roles_map(docs)
    llm = get_provider(settings, fake=fake)
    out_path = Path(out_dir) if out_dir else Path(settings.eval_out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for size in chunk_sizes:
        size_settings = settings.replace(chunk_size=int(size))
        chunks = chunk_corpus(size_settings, docs)
        embedder = FakeEmbedder(seed=settings.seed) if fake \
            else build_embedder(settings)
        store, bm25 = build_index_in_memory(size_settings, chunks, embedder)
        reranker = FakeReranker() if fake else build_reranker(settings)
        for mode in modes:
            pipeline = RAGPipeline(
                settings=size_settings, embedder=embedder,
                dense=DenseRetriever(embedder, store),
                bm25=BM25Retriever(bm25), generator=llm, reranker=reranker)
            rows: list[dict[str, Any]] = []
            for item in items:
                response = pipeline.ask(
                    item["question"], mode=mode, k=4,
                    user_role=item.get("user_role", "guest"))
                rows.append(evaluate_item(
                    item, response, embedder, roles,
                    settings.groundedness_threshold))
            agg = aggregate(rows)
            results.append({"chunk_size": int(size), "mode": mode, **agg})
            print(f"chunk_size={size} mode={mode}: "
                  f"{agg['n_items']} items, recall@4={agg['recall@4']}, "
                  f"refusal={agg['refusal_accuracy']}")
    write_csv(out_path / "results.csv", results)
    write_md(out_path / "results.md", results, fake=fake)
    return results


def write_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def write_md(path: Path, results: Sequence[dict[str, Any]],
             fake: bool) -> None:
    lines = [
        "# Retrieval evaluation results",
        "",
        ("**These numbers are APPROXIMATE single-machine measurements from "
         "this run on this corpus.**" if not fake else
         "**HARNESS SMOKE RUN with FakeEmbedder + FakeLLM (--fake): these "
         "are NOT quality results, only proof that the sweep executes.**"),
        "",
        "| " + " | ".join(CSV_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(CSV_COLUMNS)) + "|",
    ]
    for row in results:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in CSV_COLUMNS) + " |")
    lines += [
        "",
        "Metrics: recall/MRR/nDCG are computed over gold source documents at "
        "k=4; groundedness is the fraction of answer sentences with cosine "
        ">= {:.2f} to a retrieved chunk (a proxy, see rag/evaluation/"
        "metrics.py); refusal accuracy covers out-of-scope items; injection "
        "resistance is 1 when no canary/prompt leak appears; leak rate must "
        "be 0 (access-control items); citation validity uses the "
        "valid/max(1,total) convention (uncited answers count 1.0)."
        .format(0.6),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate chunk sizes x retrieval modes on the eval set.")
    parser.add_argument("--fake", action="store_true",
                        help="FakeEmbedder + FakeLLM (harness smoke test only)")
    parser.add_argument("--provider", default=None,
                        help="Override LLM provider (ollama|openai|azure|fake)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N items")
    parser.add_argument("--chunk-sizes", nargs="+", type=int,
                        default=list(DEFAULT_CHUNK_SIZES))
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args(argv)

    from rag.config import Settings

    settings = Settings.from_env()
    if args.provider:
        settings = settings.replace(llm_provider=args.provider)
    results = run(settings, fake=args.fake, limit=args.limit,
                  chunk_sizes=args.chunk_sizes, modes=args.modes,
                  out_dir=args.out)
    print(f"wrote {len(results)} result rows "
          f"to {args.out or settings.eval_out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
