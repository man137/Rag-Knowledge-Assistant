"""Tests for observability.logging_setup and observability.metrics_store."""

from __future__ import annotations

import io
import json
import logging
import sqlite3

import pytest

from observability.logging_setup import (
    JsonFormatter,
    configure_logging,
    set_request_id,
)
from observability.metrics_store import MetricsStore

# ------------------------- logging ------------------------- #

def _capture_logger():
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("tests.jsonlogs")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger, buf


def test_json_format_parses_and_redacts_pii():
    logger, buf = _capture_logger()
    logger.warning("customer wrote from jane.doe@example.com about refunds")
    payload = json.loads(buf.getvalue())
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "tests.jsonlogs"
    assert "jane.doe@example.com" not in payload["msg"]
    assert "[REDACTED-EMAIL]" in payload["msg"]
    assert "ts" in payload and "request_id" in payload


def test_request_id_propagation():
    logger, buf = _capture_logger()
    set_request_id("req-log-1")
    try:
        logger.info("hello")
        assert json.loads(buf.getvalue())["request_id"] == "req-log-1"
    finally:
        set_request_id(None)
    buf.seek(0)
    buf.truncate()
    logger.info("hello again")
    assert json.loads(buf.getvalue())["request_id"] is None


def test_configure_logging_idempotent():
    configure_logging("INFO")
    root1 = list(logging.getLogger().handlers)
    configure_logging("INFO")
    root2 = list(logging.getLogger().handlers)
    assert len(root2) == len(root1)   # no handler stacking


# ------------------------- metrics store ------------------------- #

def _record(store, request_id="r-1", mode="hybrid", latency=100.0,
            ttft=12.0, no_answer=0, invalid=0, prompt_version="v2",
            tokens_out=40):
    store.record_query(request_id=request_id, question_hash="abc123def4567890",
                       mode=mode, prompt_version=prompt_version, user_role="guest",
                       latency_ms=latency, ttft_ms=ttft, tokens_in=300,
                       tokens_out=tokens_out, no_answer=no_answer,
                       invalid_citations=invalid)


def test_record_and_rows_never_store_question_text(settings):
    store = MetricsStore(settings.metrics_db_path)
    _record(store)
    rows = store.rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["request_id"] == "r-1"
    assert row["question_hash"] == "abc123def4567890"
    assert row["mode"] == "hybrid" and row["user_role"] == "guest"
    assert "question" not in row        # no raw question column at all
    store.close()


def test_summary_values(settings):
    store = MetricsStore(settings.metrics_db_path)
    for i in range(10):
        _record(store, request_id=f"r{i}", latency=10.0 * (i + 1),
                mode="dense" if i % 2 else "hybrid",
                no_answer=1 if i >= 8 else 0,
                invalid=1 if i == 9 else 0)
    _record(store, request_id="fb-1")     # dense row, answer row
    _record(store, request_id="fb-2", no_answer=1)
    store.record_feedback("fb-1", "up")
    store.record_feedback("fb-2", "down")
    summary = store.summary()
    assert summary["total_queries"] == 12
    assert summary["no_answer_rate"] == round(3 / 12, 4)
    assert summary["latency_p50_ms"] > 0
    assert summary["latency_p95_ms"] >= summary["latency_p50_ms"]
    assert summary["invalid_citation_rate"] == round(1 / 12, 4)
    assert summary["feedback"] == {"up": 1, "down": 1, "ratio": 0.5}
    assert set(summary["by_mode"]) == {"dense", "hybrid"}
    assert summary["by_mode"]["dense"]["queries"] == 5   # odd i in 0..9
    assert summary["by_mode"]["hybrid"]["queries"] == 7  # even i + 2 extra
    store.close()


def test_feedback_unknown_request_id(settings):
    store = MetricsStore(settings.metrics_db_path)
    _record(store)
    assert store.record_feedback("missing-id", "up") is False
    assert store.record_feedback("r-1", "up") is True
    assert store.rows()[0]["feedback"] == "up"
    with pytest.raises(ValueError):
        store.record_feedback("r-1", "sideways")
    store.close()


def test_empty_summary_shape(settings):
    store = MetricsStore(settings.metrics_db_path)
    assert store.summary() == {
        "total_queries": 0, "no_answer_rate": None,
        "latency_p50_ms": None, "latency_p95_ms": None,
        "invalid_citation_rate": None,
        "feedback": {"up": 0, "down": 0, "ratio": None}, "by_mode": {}}
    store.close()


def test_schema_matches_sql_file(settings):
    """The store's auto-created schema must agree with sql/schema.sql."""
    store = MetricsStore(settings.metrics_db_path)
    _record(store)
    store.close()
    conn = sqlite3.connect(settings.metrics_db_path)
    try:
        named = {row[1] for row in conn.execute(
            "PRAGMA table_info(query_log)")}
    finally:
        conn.close()
    sql_text = (settings.metrics_db_path and open(  # noqa: SIM102
        "sql/schema.sql", encoding="utf-8").read().lower())
    for column in ("ts", "request_id", "question_hash", "mode",
                   "prompt_version", "user_role", "latency_ms", "ttft_ms",
                   "tokens_in", "tokens_out", "no_answer",
                   "invalid_citations", "feedback"):
        assert column in named
        assert column in sql_text
