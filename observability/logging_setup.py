"""JSON logging, request-id propagation, PII redaction.

WHAT: configure_logging() installs a JSON formatter on the root logger
(stdout, optional file); a contextvar carries the request id so every log
line inside a request is correlated; the formatter runs rag.security
redact_pii() over every message so emails/phones/cards never reach the logs.
WHY: grep-able structured logs with a request id are the minimum observability
for an API that will be handed to someone else to operate; PII redaction at
the formatter level is a data-handling control, not a developer guideline.
MATHS: none.
FAILURE MODE: contextvars are per-task — a bare-thread pool that does not
propagate context loses the request id (logs still work, with a null id).
Redaction is regex-based and documented as incomplete: it stops obvious PII,
not paraphrases or free-text descriptions.
INTERVIEW ONE-LINER: "One JSON line per event with a contextvar request-id
and regex PII redaction in the formatter — the same shape as Application
Insights telemetry, just local."
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

from rag.security import redact_pii

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None)

_CONFIGURED = False


def set_request_id(request_id: str | None) -> None:
    """Bind the current request id for all log lines in this task."""
    _request_id_var.set(request_id)


def get_request_id() -> str | None:
    return _request_id_var.get()


class JsonFormatter(logging.Formatter):
    """One JSON object per record: ts, level, logger, msg, request_id (+exc)."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        message, _redacted_count = redact_pii(message)   # never log raw PII
        payload = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": message,
            "request_id": get_request_id(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO",
                      log_file: str | Path | None = None) -> None:
    """Idempotent: reconfiguration replaces handlers instead of stacking."""
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    formatter = JsonFormatter()
    if _CONFIGURED:
        for handler in list(root.handlers):
            handler.setFormatter(formatter)
        return
    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    _CONFIGURED = True
