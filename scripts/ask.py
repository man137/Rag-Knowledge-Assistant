#!/usr/bin/env python3
"""Ask CLI: question in, grounded answer out (offline-friendly).

WHAT: `python scripts/ask.py [--fake] "question" [--mode hybrid]
[--role guest] [--k 4] [--prompt-version v2] [--json]` loads the saved index
(built by scripts/ingest.py), runs the full pipeline and prints the answer
with its sources, timings, citation validity and flags. With --fake and no
saved index it builds a temporary in-memory fake index from data/docs.
WHY: the fastest way to smoke-test the whole stack end to end, and the tool
used in the README's ingest -> ask verification step.
MATHS: none.
FAILURE MODE: with a real embedder there must be a saved index first
(ingest without --fake); otherwise the command says so and exits non-zero.
INTERVIEW ONE-LINER: "One command answers from the index with citations,
per-stage timings and ACL enforcement — the same pipeline object the API
uses."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.main import load_or_build_pipeline  # noqa: E402
from observability.logging_setup import configure_logging  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.pipeline import PIPELINE_MODES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("--fake", action="store_true",
                        help="FakeEmbedder + FakeLLM (offline smoke test)")
    parser.add_argument("--mode", choices=PIPELINE_MODES, default="hybrid")
    parser.add_argument("--role", default="guest",
                        help="user role (guest|staff|manager|hr)")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--json", action="store_true",
                        help="print the full response as JSON")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.fake:
        settings = settings.replace(llm_provider="fake")
    configure_logging(settings.log_level)
    settings.apply_seed()

    try:
        pipeline = load_or_build_pipeline(settings)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run  python scripts/ingest.py  first.")
        return 1

    response = pipeline.ask(args.question, mode=args.mode, k=args.k,
                            user_role=args.role,
                            prompt_version=args.prompt_version)
    if args.json:
        print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print(f"ANSWER:\n{response.answer}\n")
    print("SOURCES:")
    for source in response.sources:
        name = Path(source.source).name
        print(f"  - [{source.chunk_id}] {name} (score={source.score:.4f}) "
              f"ranks={source.ranks}")
        print(f"      {source.snippet[:120]}")
    print(f"\nno_answer={response.no_answer}  "
          f"invalid_citations={response.invalid_citations}  "
          f"injection_flags={len(response.injection_flags)}")
    print("timings (ms):", {k: round(v, 2) for k, v in response.timings.items()},
          f"ttft={response.ttft_ms:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
