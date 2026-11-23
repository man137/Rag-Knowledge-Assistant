"""Validate data/eval/eval_set.jsonl: schema, counts, gold-keyword presence.

WHAT: checks the eval set line by line: valid JSON objects, all required
fields with the right types, ids unique, roles valid, the exact type counts
(30 in_scope / 8 out_of_scope / 6 injection / 6 access_control), every gold
keyword actually present in its gold source document, injection items
carrying the canary in notes and including at least two docs-side probes,
and access_control items naming a document genuinely restricted for that
role (checked against the front matter of the real corpus files).
WHY: an eval set with a typo'd gold keyword silently measures the wrong
thing; this script runs in CI and fails the build on any drift between the
questions, the documents and the expected counts.
MATHS: none.
FAILURE MODE: the keyword check is exact substring matching after
lowercasing on both sides — a gold keyword that is a paraphrase (not a
literal document phrase) is reported as an error, by design, so keywords
stay verifiable.
INTERVIEW ONE-LINER: "The eval set is itself tested: CI verifies every gold
keyword literally appears in the gold source, counts are exact, and
access-control items really are restricted for the role under test."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag.loaders import load_directory  # noqa: E402
from rag.security import ROLES  # noqa: E402

REQUIRED_FIELDS = ("id", "type", "question", "user_role", "gold_sources",
                   "gold_keywords", "must_refuse", "notes")
EXPECTED_COUNTS = {"in_scope": 30, "out_of_scope": 8, "injection": 6,
                   "access_control": 6}
EVAL_TYPES = tuple(EXPECTED_COUNTS)
CANARY = "SECRET-CANARY-8F42"


def validate(eval_set_path: str | Path,
             docs_dir: str | Path | None = None) -> tuple[list[str], dict[str, Any]]:
    """Returns (errors, summary). errors == [] means the eval set is valid."""
    errors: list[str] = []
    eval_set_path = Path(eval_set_path)
    if docs_dir is None:
        docs_dir = _REPO_ROOT / "data" / "docs"
    docs = load_directory(docs_dir)
    by_name: dict[str, Any] = {Path(d.source).name: d for d in docs}

    items: list[dict[str, Any]] = []
    try:
        raw_lines = eval_set_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"cannot read eval set: {exc}"], {"items": 0}
    for lineno, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON ({exc})")
            continue
        items.append(item)

    ids: set[str] = set()
    counts = dict.fromkeys(EVAL_TYPES, 0)
    for index, item in enumerate(items, start=1):
        where = f"item {index} ({item.get('id', '?')})"
        for field in REQUIRED_FIELDS:
            if field not in item:
                errors.append(f"{where}: missing field {field!r}")
        if "id" in item:
            if item["id"] in ids:
                errors.append(f"{where}: duplicate id {item['id']!r}")
            ids.add(item["id"])
        if item.get("type") not in EVAL_TYPES:
            errors.append(f"{where}: type {item.get('type')!r} not in {EVAL_TYPES}")
            continue
        counts[item["type"]] += 1
        if item.get("user_role") not in ROLES:
            errors.append(f"{where}: user_role {item.get('user_role')!r} not in {ROLES}")
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            errors.append(f"{where}: question must be a non-empty string")
        if not isinstance(item.get("gold_sources"), list):
            errors.append(f"{where}: gold_sources must be a list")
        if not isinstance(item.get("gold_keywords"), list):
            errors.append(f"{where}: gold_keywords must be a list")
        if not isinstance(item.get("must_refuse"), bool):
            errors.append(f"{where}: must_refuse must be a bool")

        if item["type"] == "in_scope":
            if item.get("must_refuse") is not False:
                errors.append(f"{where}: in_scope items must have must_refuse=false")
            if not item.get("gold_sources"):
                errors.append(f"{where}: in_scope items need gold_sources")
            for keyword in item.get("gold_keywords", []):
                found_in_any = False
                for source in item.get("gold_sources", []):
                    doc = by_name.get(source)
                    if doc is None:
                        errors.append(f"{where}: gold source {source!r} "
                                      "is not in data/docs")
                        continue
                    if str(keyword).lower() in doc.text.lower():
                        found_in_any = True
                if not found_in_any:
                    errors.append(f"{where}: gold keyword {keyword!r} not "
                                  "found in any gold source document")
        elif item["type"] == "out_of_scope":
            if item.get("must_refuse") is not True:
                errors.append(f"{where}: out_of_scope items must refuse")
            if item.get("gold_sources"):
                errors.append(f"{where}: out_of_scope items must have no "
                              "gold_sources")
        elif item["type"] == "injection":
            if item.get("must_refuse") is not True:
                errors.append(f"{where}: injection items must refuse")
            if CANARY not in str(item.get("notes", "")):
                errors.append(f"{where}: injection items must carry the "
                              f"canary {CANARY!r} in notes")
        elif item["type"] == "access_control":
            if item.get("must_refuse") is not True:
                errors.append(f"{where}: access_control items must refuse")
            if not item.get("gold_sources"):
                errors.append(f"{where}: access_control items need the "
                              "restricted gold source")
            role = item.get("user_role")
            for source in item.get("gold_sources", []):
                doc = by_name.get(source)
                if doc is None:
                    errors.append(f"{where}: gold source {source!r} "
                                  "is not in data/docs")
                    continue
                permitted = {"all", role} if role in ROLES else {"all", "guest"}
                if set(doc.allowed_roles).intersection(permitted):
                    errors.append(
                        f"{where}: {source} is READABLE by role {role!r}; "
                        "access_control probes need a restricted document")

    for type_name, expected in EXPECTED_COUNTS.items():
        if counts[type_name] != expected:
            errors.append(f"expected {expected} {type_name} items, "
                         f"found {counts[type_name]}")
    docs_side = [i for i in items if i["type"] == "injection"
                 and i.get("gold_sources")]
    if len(docs_side) < 2:
        errors.append("need at least 2 docs-side injection items "
                      "(gold_sources set)")
    summary = {"items": len(items), "counts": counts,
               "docs_side_injection": len(docs_side),
               "errors": len(errors)}
    return errors, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the evaluation set against the synthetic corpus.")
    parser.add_argument("--eval-set", default=str(
        _REPO_ROOT / "data" / "eval" / "eval_set.jsonl"))
    parser.add_argument("--docs-dir", default=str(_REPO_ROOT / "data" / "docs"))
    args = parser.parse_args(argv)
    errors, summary = validate(args.eval_set, args.docs_dir)
    print(f"eval set: {summary['items']} items, counts={summary['counts']}, "
          f"docs-side injection probes={summary['docs_side_injection']}")
    if errors:
        print(f"FAILED with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("OK: schema, counts, gold keywords and access-control restrictions "
          "all verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
