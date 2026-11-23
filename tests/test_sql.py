"""Tests: the 6 analytics queries run against a seeded SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from observability.metrics_store import MetricsStore

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def _seed(db_path):
    store = MetricsStore(db_path)
    for i in range(20):
        store.record_query(
            request_id=f"r{i:02d}",
            question_hash=f"hash{i:016x}"[:16],
            mode=["dense", "bm25", "hybrid", "hybrid_rerank"][i % 4],
            prompt_version=["v1", "v2", "v3"][i % 3],
            user_role=["guest", "staff", "manager", "hr"][i % 4],
            latency_ms=float(50 + i * 7),
            ttft_ms=float(10 + i),
            tokens_in=300 + i, tokens_out=40,
            no_answer=1 if i % 5 == 0 else 0,
            invalid_citations=1 if i % 7 == 0 else 0)
    for rid in ("r00", "r01", "r02", "r10", "r11"):
        store.record_feedback(rid, "up" if rid.endswith(("0", "1")) else "down")
    store.close()


def _statements(sql_text: str) -> list[str]:
    parts = [part.strip() for part in sql_text.split(";")]
    return [part for part in parts if part]


def test_schema_sql_creates_the_same_table(tmp_path):
    db = tmp_path / "fresh.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript((SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
        columns = {row[1] for row in conn.execute("PRAGMA table_info(query_log)")}
        assert {"ts", "request_id", "question_hash", "mode", "feedback"} <= columns
    finally:
        conn.close()


def test_all_six_analytics_queries_run_and_return_rows(tmp_path):
    db = tmp_path / "seeded.db"
    _seed(db)
    analytics = (SQL_DIR / "analytics.sql").read_text(encoding="utf-8")
    statements = _statements(analytics)
    # 6 queries, each ending with a semicolon before the next -- comment
    assert len(statements) == 6
    conn = sqlite3.connect(db)
    try:
        for index, statement in enumerate(statements, start=1):
            rows = conn.execute(statement).fetchall()
            assert len(rows) >= 1, f"query {index} returned no rows"
    finally:
        conn.close()


def test_percentile_query_orders_p50_below_p95(tmp_path):
    db = tmp_path / "seeded.db"
    _seed(db)
    statement = _statements(
        (SQL_DIR / "analytics.sql").read_text(encoding="utf-8"))[0]
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(statement).fetchall()
        for _mode, queries, p50, p95 in rows:
            assert queries > 0
            assert p50 is not None and p95 is not None
            assert p50 <= p95
    finally:
        conn.close()


def test_no_answer_rate_and_daily_volume_queries(tmp_path):
    db = tmp_path / "seeded.db"
    _seed(db)
    statements = _statements(
        (SQL_DIR / "analytics.sql").read_text(encoding="utf-8"))
    conn = sqlite3.connect(db)
    try:
        for _day, queries, rate in conn.execute(statements[1]).fetchall():
            assert queries == 20            # single seeded day
            assert 0.0 <= rate <= 1.0
            assert rate == 0.2              # 4 no-answer rows of 20
        volume = conn.execute(statements[5]).fetchall()
        assert volume[0][1] == 20
    finally:
        conn.close()
