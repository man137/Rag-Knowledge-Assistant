#!/usr/bin/env python3
"""Export the OpenAPI spec of the API for the Copilot Studio connector.

WHAT: writes openapi.json (the full FastAPI OpenAPI 3 schema) so it can be
imported as a custom connector in Power Platform / Copilot Studio.
WHY: step 2 of COPILOT_STUDIO_GUIDE.md needs the machine-readable contract
of /ask, /ask_sync, /feedback, /metrics and /health.
MATHS: none.
FAILURE MODE: the exported spec contains no auth requirement for the
connector import — API-key authentication is configured in the connector
wizard (see the guide), not in the file itself.
INTERVIEW ONE-LINER: "The connector is driven by the exported openapi.json,
so the API contract stays the single source of truth."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.main import create_app  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(_REPO_ROOT / "openapi.json"))
    args = parser.parse_args(argv)
    # a Settings-scoped app (no pipeline build) is enough for the schema
    app = create_app()
    spec = app.openapi()
    Path(args.out).write_text(json.dumps(spec, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    paths = ", ".join(sorted(spec.get("paths", {})))
    print(f"wrote {args.out} ({len(spec.get('paths', {}))} paths: {paths})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
