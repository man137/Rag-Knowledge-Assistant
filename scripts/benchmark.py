#!/usr/bin/env python3
"""Benchmark: IVF vs brute-force vector search (recall@10 and latency).

WHAT: `python scripts/benchmark.py [--n 5000] [--dim 64] [--nlist 16]
[--probes 1 4 16] [--repeats 200] [--docs]` generates seeded random unit
vectors (or, with --docs, embeds the real synthetic corpus with the fake
embedder), builds the brute-force VectorStore and the IVFIndex, measures
recall@10 and average per-query latency for each nprobe, and prints a table.
The printed numbers are whatever this run produced on this machine.
WHY: ANN indexes exist to trade a little recall for a lot of speed; the
benchmark shows that trade-off with real measurements instead of folklore.
MATHS: recall@10 = mean over queries of |top10(IVF) ∩ top10(brute)| / 10.
Latency is wall-clock per query averaged over --repeats queries (sensitive to
machine load; report the machine alongside any number you quote).
FAILURE MODE: at this corpus scale brute force is already microseconds, so
IVF "wins" only on much larger corpora — the table makes the crossover
visible instead of hiding it.
INTERVIEW ONE-LINER: "I hand-wrote IVF (seeded k-means + inverted lists)
and measured recall@10 against brute force at several nprobe values — the
same trade-off Azure AI Search's ANN index makes for you."
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag.chunking import Chunk  # noqa: E402
from rag.embedder import FakeEmbedder  # noqa: E402
from rag.loaders import load_directory  # noqa: E402
from rag.pipeline import chunk_corpus  # noqa: E402
from rag.vector_store import IVFIndex, VectorStore  # noqa: E402


def _corpus(args) -> tuple[list[Chunk], object]:
    import numpy as np

    if args.docs:
        settings_data = _REPO_ROOT / "data" / "docs"
        from rag.config import Settings

        settings = Settings()
        docs = load_directory(settings_data)
        chunks = chunk_corpus(settings, docs)
        embedder = FakeEmbedder()
        vectors = embedder.encode_documents([c.text for c in chunks])
        return chunks, vectors
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((args.n, args.dim))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    chunks = [Chunk(chunk_id=f"c{i:06d}", text=f"synthetic {i}", source="bench.md",
                    title="bench", page=1, start_char=0, end_char=10,
                    token_count=2, allowed_roles=["all"])
              for i in range(args.n)]
    return chunks, vectors.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=5000,
                        help="synthetic corpus size (ignored with --docs)")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--nlist", type=int, default=16)
    parser.add_argument("--probes", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--docs", action="store_true",
                        help="use the real synthetic corpus with FakeEmbedder")
    args = parser.parse_args(argv)

    import numpy as np

    chunks, vectors = _corpus(args)
    n = len(chunks)
    print(f"corpus: {n} vectors, dim={vectors.shape[1]}")

    store = VectorStore(dim=vectors.shape[1])
    store.add(chunks, vectors)

    rng = np.random.default_rng(7)
    queries = rng.standard_normal((args.repeats, vectors.shape[1]))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # brute force reference timing
    start = time.perf_counter()
    for q in queries:
        store.search(q, 10)
    brute_ms = (time.perf_counter() - start) * 1000 / args.repeats

    ivf = IVFIndex(nlist=args.nlist, seed=42)
    build_start = time.perf_counter()
    ivf.build(chunks, vectors)
    build_ms = (time.perf_counter() - build_start) * 1000

    print(f"\nbrute force: {brute_ms:.4f} ms/query (avg of {args.repeats})")
    print(f"IVF build (k-means, nlist={args.nlist}): {build_ms:.1f} ms\n")
    header = (f"{'nprobe':>8} {'recall@10':>12} {'avg ms/query':>14} "
              f"{'vs brute':>10}")
    print(header)
    print("-" * len(header))
    for probe in args.probes:
        recall = ivf.recall_at_k(queries, 10, store)
        start = time.perf_counter()
        for q in queries:
            ivf.search(q, 10, nprobe=probe)
        probe_ms = (time.perf_counter() - start) * 1000 / args.repeats
        ratio = f"{probe_ms / brute_ms:.2f}x"
        print(f"{probe:>8} {recall:>12.4f} {probe_ms:>14.4f} {ratio:>10}")
    print("\nNOTE: numbers are measurements from this single run on this "
          "machine; treat them as approximate and re-measure before quoting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
