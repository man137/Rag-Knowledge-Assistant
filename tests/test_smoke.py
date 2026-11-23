"""End-to-end CLI smoke tests: ingest -> ask, agent demo, benchmark, export.

These run the documented commands as subprocesses, exactly as the README
tells the user to. All use --fake / fake defaults, so everything is offline.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *cmd], cwd=REPO,
                          capture_output=True, text=True, timeout=timeout,
                          check=False)


def test_ingest_then_ask_cli_end_to_end():
    ingest = _run(["scripts/ingest.py", "--fake"])
    assert ingest.returncode == 0, ingest.stdout + ingest.stderr
    assert "saved vector store" in ingest.stdout
    assert "fake-embedder" in ingest.stdout

    ask = _run(["scripts/ask.py", "--fake",
                "What is the refund rule for event cancellation?"])
    assert ask.returncode == 0, ask.stdout + ask.stderr
    assert "ANSWER:" in ask.stdout
    assert "event_registration_faq.md" in ask.stdout          # right source
    assert "no_answer=False" in ask.stdout.replace(" ", "")


def test_agent_demo_cli_shows_full_trace():
    demo = _run(["scripts/run_agent_demo.py"])
    assert demo.returncode == 0, demo.stdout + demo.stderr
    out = demo.stdout
    assert "SCENARIO A" in out and "SCENARIO B" in out
    assert "thought:" in out and "action=" in out and "observation:" in out
    assert "#support-escalations" in out          # Teams escalation happened
    assert "CRM tickets created this run: 1" in out
    assert "FINAL ANSWER" in out
    assert "[source:" in out                      # KB answer carries citations


def test_export_openapi_cli(tmp_path):
    out_file = tmp_path / "openapi.json"
    export = _run(["scripts/export_openapi.py", "--out", str(out_file)])
    assert export.returncode == 0, export.stdout + export.stderr
    spec = json.loads(out_file.read_text(encoding="utf-8"))
    for path in ("/ask", "/ask_sync", "/ingest", "/feedback", "/metrics",
                 "/health"):
        assert path in spec["paths"], f"missing {path} in openapi.json"
    assert spec["paths"]["/ask"]["post"]["summary"].startswith("Ask")


def test_benchmark_cli_prints_recall_table():
    bench = _run(["scripts/benchmark.py", "--n", "600", "--repeats", "40",
                  "--probes", "1", "8", "--nlist", "8"])
    assert bench.returncode == 0, bench.stdout + bench.stderr
    assert "recall@10" in bench.stdout
    assert "nprobe" in bench.stdout
    assert "approximate" in bench.stdout


def test_validate_eval_set_cli():
    result = _run(["scripts/validate_eval_set.py"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: schema, counts" in result.stdout
