"""SQLite metrics store: one row per ask(), never the raw question.

WHAT: MetricsStore records query_log rows (ts, request_id, question_hash, mode,
prompt_version, user_role, latency_ms, ttft_ms, tokens_in, tokens_out,
no_answer, invalid_citations, feedback) into SQLite, supports thumbs up/down
feedback by request_id, and computes a summary (counts, no-answer rate, p50/p95
latency overall and by mode, feedback and invalid-citation rates).
WHY: "evaluation of accuracy / latency / response quality" and "logging and
monitoring" need recorded facts; SQLite keeps the project dependency-free
while sql/analytics.sql shows how the same data would be queried in
production.
MATHS: the raw question text is never stored — only sha1(question)[:16] — so
repeated questions can be grouped for analytics without keeping user content.
Percentiles use linear interpolation between closest ranks (see
rag/evaluation/metrics.py); the SQL analytics file uses nearest-rank, which
approximates it — both are documented.
FAILURE MODE: SQLite is single-writer; the store serialises writes with a
lock and check_same_thread=False, which is fine for a demo and would be
replaced by a managed database in production. Feedback for an unknown
request_id updates zero rows and is reported as such (not an exception).
INTERVIEW ONE-LINER: "Metrics go to SQLite with the question hashed, never
stored raw — latency, no-answer rate and feedback are queryable per mode,
which is exactly the data an SLA review asks for."
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
import threading
from pathlib import Path
from typing import Any

from rag.evaluation.metrics import percentile

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT NOT NULL,
    question_hash TEXT NOT NULL,
    mode TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    user_role TEXT NOT NULL,
    latency_ms REAL NOT NULL DEFAULT 0,
    ttft_ms REAL NOT NULL DEFAULT 0,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    no_answer INTEGER NOT NULL DEFAULT 0,
    invalid_citations INTEGER NOT NULL DEFAULT 0,
    feedback TEXT
);
CREATE INDEX IF NOT EXISTS idx_query_log_mode ON query_log(mode);
CREATE INDEX IF NOT EXISTS idx_query_log_ts ON query_log(ts);
CREATE INDEX IF NOT EXISTS idx_query_log_request_id ON query_log(request_id);
"""

FEEDBACK_VALUES = ("up", "down")


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


class MetricsStore:
    """Thread-safe wrapper around one SQLite metrics database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record_query(self, *, request_id: str, question_hash: str,
                     mode: str, prompt_version: str, user_role: str,
                     latency_ms: float, ttft_ms: float,
                     tokens_in: int, tokens_out: int, no_answer: int,
                     invalid_citations: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO query_log (ts, request_id, question_hash, mode,"
                " prompt_version, user_role, latency_ms, ttft_ms, tokens_in,"
                " tokens_out, no_answer, invalid_citations, feedback)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (_now_iso(), str(request_id), str(question_hash), str(mode),
                 str(prompt_version), str(user_role), float(latency_ms),
                 float(ttft_ms), int(tokens_in), int(tokens_out),
                 int(no_answer), int(invalid_citations)))
            self._conn.commit()

    def record_feedback(self, request_id: str, value: str) -> bool:
        """Attach thumbs up/down to a request. Returns False when the
        request_id is unknown (no row updated)."""
        if value not in FEEDBACK_VALUES:
            raise ValueError(f"feedback must be one of {FEEDBACK_VALUES}, "
                             f"got {value!r}")
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE query_log SET feedback = ? WHERE request_id = ?",
                (value, str(request_id)))
            self._conn.commit()
            return cursor.rowcount > 0

    def rows(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Raw rows, newest first (for tests and debugging)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (limit,))
            names = [d[0] for d in cur.description]
            return [dict(zip(names, row, strict=True))
                    for row in cur.fetchall()]

    def summary(self) -> dict[str, Any]:
        """Aggregated health metrics computed for GET /metrics."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT mode, latency_ms, no_answer, invalid_citations,"
                " feedback FROM query_log")
            records = cur.fetchall()
        total = len(records)
        if total == 0:
            return {"total_queries": 0, "no_answer_rate": None,
                    "latency_p50_ms": None, "latency_p95_ms": None,
                    "invalid_citation_rate": None,
                    "feedback": {"up": 0, "down": 0, "ratio": None},
                    "by_mode": {}}
        modes = {row[0] for row in records}
        latencies = [float(row[1]) for row in records]
        no_answer = sum(int(row[2]) for row in records)
        invalid = sum(1 for row in records if int(row[3]) > 0)
        ups = sum(1 for row in records if row[4] == "up")
        downs = sum(1 for row in records if row[4] == "down")
        by_mode: dict[str, Any] = {}
        for mode in sorted(modes):
            mode_rows = [row for row in records if row[0] == mode]
            mode_lat = [float(r[1]) for r in mode_rows]
            by_mode[mode] = {
                "queries": len(mode_rows),
                "no_answer_rate": round(
                    sum(int(r[2]) for r in mode_rows) / len(mode_rows), 4),
                "latency_p50_ms": round(percentile(mode_lat, 50), 3),
                "latency_p95_ms": round(percentile(mode_lat, 95), 3),
            }
        ratio = round(ups / (ups + downs), 4) if (ups + downs) else None
        return {
            "total_queries": total,
            "no_answer_rate": round(no_answer / total, 4),
            "latency_p50_ms": round(percentile(latencies, 50), 3),
            "latency_p95_ms": round(percentile(latencies, 95), 3),
            "invalid_citation_rate": round(invalid / total, 4),
            "feedback": {"up": ups, "down": downs, "ratio": ratio},
            "by_mode": by_mode,
        }
