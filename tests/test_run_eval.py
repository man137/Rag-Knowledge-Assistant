"""Tests for the eval harness: run_eval sweep + prompt_ab + validator."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rag.evaluation import prompt_ab, run_eval
from scripts.validate_eval_set import validate

REPO = Path(__file__).resolve().parents[1]


def test_eval_set_passes_validation():
    errors, summary = validate(REPO / "data" / "eval" / "eval_set.jsonl",
                               REPO / "data" / "docs")
    assert errors == []
    assert summary["items"] == 50
    assert summary["counts"] == {"in_scope": 30, "out_of_scope": 8,
                                 "injection": 6, "access_control": 6}
    assert summary["docs_side_injection"] >= 2


def test_validator_catches_broken_sets(tmp_path):
    bad = [
        {"id": "x1", "type": "in_scope", "question": "q", "user_role": "guest",
         "gold_sources": ["travel_expense_policy.md"],
         "gold_keywords": ["zzz-not-in-doc-xyz"], "must_refuse": False,
         "notes": ""},
        {"id": "x2", "type": "access_control", "question": "q",
         "user_role": "hr", "gold_sources": ["hr_compensation_leave_policy.md"],
         "gold_keywords": [], "must_refuse": True, "notes": "hr can read this"},
    ]
    path = tmp_path / "bad_eval.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in bad), encoding="utf-8")
    errors, summary = validate(path, REPO / "data" / "docs")
    assert any("gold keyword" in e for e in errors)
    assert any("READABLE by role" in e for e in errors)  # hr can read hr doc
    assert any("expected 30 in_scope" in e for e in errors)


def test_run_eval_fake_smoke(settings, tmp_path):
    """Harness check only: FakeEmbedder/FakeLLM prove the sweep runs and the
    outputs land on disk — these are NOT quality measurements. Uses the full
    50-item set so every metric bucket (including injection resistance and
    leak rate) is populated."""
    out = tmp_path / "eval_out"
    results = run_eval.run(settings, fake=True, limit=None,
                           chunk_sizes=(256,), modes=("hybrid",),
                           out_dir=out)
    assert len(results) == 1
    row = results[0]
    assert row["chunk_size"] == 256 and row["mode"] == "hybrid"
    assert row["n_items"] == 50
    for key in ("recall@4", "groundedness", "citation_validity",
                "refusal_accuracy", "injection_resistance", "leak_rate",
                "latency_p50_ms", "latency_p95_ms", "ttft_p50_ms"):
        assert row[key] is not None, f"{key} missing from results"
    # the ACL guarantees zero leakage whatever the embedder
    assert row["leak_rate"] == 0.0
    csv_path = out / "results.csv"
    md_path = out / "results.md"
    assert csv_path.is_file() and md_path.is_file()
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["chunk_size"] == "256"
    assert "HARNESS SMOKE RUN" in md_path.read_text(encoding="utf-8")


def test_run_eval_full_sweep_shape(settings, tmp_path):
    out = tmp_path / "eval_out"
    results = run_eval.run(settings, fake=True, limit=6,
                            chunk_sizes=(256, 512), modes=("dense", "bm25"),
                            out_dir=out)
    assert len(results) == 4  # 2 chunk sizes x 2 modes
    assert {(r["chunk_size"], r["mode"]) for r in results} == {
        (256, "dense"), (256, "bm25"), (512, "dense"), (512, "bm25")}


def test_prompt_ab_fake_smoke(settings, tmp_path):
    out = tmp_path / "ab_out"
    results = prompt_ab.run(settings, fake=True, limit=12, out_dir=out)
    assert [r["prompt_version"] for r in results] == ["v1", "v2", "v3"]
    for row in results:
        assert row["n_items"] == 12
        assert row["refusal_accuracy"] is not None
    md = (out / "prompt_ab.md").read_text(encoding="utf-8")
    assert "| v1 |" in md and "| v2 |" in md and "| v3 |" in md
    assert "HARNESS SMOKE RUN" in md


def test_run_eval_cli_writes_outputs(settings, tmp_path, monkeypatch, capsys):
    """The documented CLI entry point runs end to end with --fake."""
    monkeypatch.setenv("EVAL_OUT_DIR", str(tmp_path / "cli_out"))
    code = run_eval.main(["--fake", "--limit", "5",
                          "--chunk-sizes", "512", "--modes", "hybrid"])
    assert code == 0
    assert (tmp_path / "cli_out" / "results.csv").is_file()
    captured = capsys.readouterr()
    assert "wrote 1 result rows" in captured.out
