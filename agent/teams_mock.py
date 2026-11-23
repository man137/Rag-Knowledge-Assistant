"""In-memory Teams mock that appends to a log file.

WHAT: post(channel, text) records the message in memory AND appends a line to
a log file so the demo has a durable trace of what a Teams post would have
sent. A MOCK: it never calls Microsoft Teams (the real webhook lives in
azure_adapters/teams_webhook.py and is NOT TESTED AGAINST LIVE SERVICE).
WHY: the escalation use case posts to #support-escalations; the mock keeps
that step executable offline while preserving the interface the real webhook
adapter implements.
MATHS: none.
FAILURE MODE: messages live in process memory plus one file; there is no
Teams delivery, no retry, no formatting. The file append is synchronous.
INTERVIEW ONE-LINER: "The Teams mock writes the exact payload the incoming
webhook would receive, so swapping in the real HTTP call is a one-line change."
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path


class TeamsMock:
    """Records Teams messages in memory and to a JSON-lines log file."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.messages: list[dict[str, str]] = []

    def post(self, channel: str, text: str) -> dict[str, str]:
        channel = (channel or "").strip() or "#general"
        text = (text or "").strip()
        if not text:
            raise ValueError("message text must not be empty")
        message = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="milliseconds"),
            "channel": channel,
            "text": text,
        }
        self.messages.append(message)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(message, ensure_ascii=False) + "\n")
        return message

    def all_messages(self) -> list[dict[str, str]]:
        return list(self.messages)
