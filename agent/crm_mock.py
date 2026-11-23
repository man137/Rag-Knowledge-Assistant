"""SQLite-backed mock CRM for the agent demo and tests.

WHAT: a tickets table (id, summary, priority, status, created_at) behind
create_ticket / get_ticket / list_tickets. A MOCK: it exists so the agent
can create CRM tickets offline; it is not a Dynamics 365 adapter (see
azure_adapters/README_AZURE.md for the production mapping).
WHY: the escalation use case needs a write target that behaves like the real
system (auto-increment ids, persisted rows) without any cloud credentials.
MATHS: none.
FAILURE MODE: SQLite file locking under heavy concurrency; fine for a demo,
replaced by the CRM client in production. Invalid priorities raise ValueError
before any insert.
INTERVIEW ONE-LINER: "The demo CRM is a SQLite table with the same interface
the production CRM adapter would implement, so the agent code never changes."
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path
from typing import Any

PRIORITIES = ("low", "medium", "high")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    summary TEXT NOT NULL,
    priority TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high')),
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);
"""


class CrmMock:
    """Minimal ticket store with the shape of a real CRM tickets module."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def create_ticket(self, summary: str, priority: str = "medium",
                      status: str = "open") -> int:
        summary = (summary or "").strip()
        if not summary:
            raise ValueError("ticket summary must not be empty")
        if priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}")
        created = _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="milliseconds")
        cursor = self._conn.execute(
            "INSERT INTO tickets (summary, priority, status, created_at)"
            " VALUES (?, ?, ?, ?)", (summary, priority, status, created))
        self._conn.commit()
        return int(cursor.lastrowid)

    def get_ticket(self, ticket_id: int) -> dict[str, Any] | None:
        cursor = self._conn.execute(
            "SELECT id, summary, priority, status, created_at FROM tickets"
            " WHERE id = ?", (int(ticket_id),))
        row = cursor.fetchone()
        if row is None:
            return None
        return {"id": row[0], "summary": row[1], "priority": row[2],
                "status": row[3], "created_at": row[4]}

    def list_tickets(self, limit: int = 20) -> list[dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT id, summary, priority, status, created_at FROM tickets"
            " ORDER BY id DESC LIMIT ?", (int(limit),))
        return [{"id": r[0], "summary": r[1], "priority": r[2],
                 "status": r[3], "created_at": r[4]} for r in cursor.fetchall()]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM tickets")
                   .fetchone()[0])
