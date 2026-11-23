#!/usr/bin/env python3
"""Ingest CLI: load docs, clean, chunk, embed, save the hybrid index.

WHAT: `python scripts/ingest.py [--fake] [--data-dir DIR] [--chunk-size N]
[--strategy fixed|sentence]` loads every .md/.txt/.pdf under the data
directory, cleans and chunks them, embeds the chunks (FakeEmbedder with
--fake, the pre-trained model otherwise), builds the BM25 index and saves
both halves under indexes/<embedder-slug>/.
WHY: indexing is an explicit, repeatable step so the serving path stays
read-only and deterministic.
MATHS: none.
FAILURE MODE: with the real model the first run downloads BAAI/bge-small-
en-v1.5 (needs network); with --fake everything is offline and the dense
no-answer threshold is recalibrated by rag.pipeline.fake_tuned.
INTERVIEW ONE-LINER: "Ingestion is one command with a --fake mode so the
whole stack runs on a laptop with zero downloads."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from observability.logging_setup import configure_logging  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.embedder import FakeEmbedder, build_embedder  # noqa: E402
from rag.loaders import load_directory  # noqa: E402
from rag.pipeline import build_index_in_memory, chunk_corpus, fake_tuned  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fake", action="store_true",
                        help="use FakeEmbedder (offline, deterministic)")
    parser.add_argument("--data-dir", default=None,
                        help="override the corpus directory")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--strategy", choices=("fixed", "sentence"),
                        default=None)
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.fake:
        settings = settings.replace(llm_provider="fake")
        settings = fake_tuned(settings)
    if args.data_dir:
        settings = settings.replace(data_dir=args.data_dir)
    if args.chunk_size:
        settings = settings.replace(chunk_size=args.chunk_size)
    if args.chunk_overlap:
        settings = settings.replace(chunk_overlap=args.chunk_overlap)
    if args.strategy:
        settings = settings.replace(chunking_strategy=args.strategy)
    settings.apply_seed()
    configure_logging(settings.log_level)

    docs = load_directory(settings.data_dir)
    if not docs:
        print(f"ERROR: no documents found under {settings.data_dir}")
        return 1
    chunks = chunk_corpus(settings, docs)
    embedder = FakeEmbedder(seed=settings.seed) if args.fake \
        else build_embedder(settings)
    print(f"loaded {len(docs)} documents -> {len(chunks)} chunks "
          f"(strategy={settings.chunking_strategy}, "
          f"size={settings.chunk_size}, overlap={settings.chunk_overlap})")

    store, bm25 = build_index_in_memory(settings, chunks, embedder)
    index_dir = settings.index_dir_for(embedder.name)
    store.save(index_dir)
    bm25.save(index_dir / "bm25.json")
    print(f"saved vector store ({store.matrix.shape}) + BM25 to {index_dir}")
    print(f"embedder: {embedder.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
