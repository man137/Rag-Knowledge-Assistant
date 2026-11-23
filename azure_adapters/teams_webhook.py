"""Teams incoming-webhook poster.

WHAT: `post_teams_webhook(url, text)` POSTs a message to a Microsoft Teams
incoming webhook or Power Automate Workflows URL with a timeout and clean
error handling — the production twin of agent.teams_mock.TeamsMock.
WHY: escalation notifications must land in a channel a human watches; this
is the one line that turns the mocked escalation into a real one.
MATHS: none.
FAILURE MODE: classic Teams incoming webhooks expect {"text": ...}; Workflows
URLs (the newer replacement) expect {"type": "message", "attachments":
[adaptive card]} — the payload shape differs per receiver, so the function
lets you pass a fully prepared payload and marks the default as
VERSION-SENSITIVE. Non-2xx responses raise with the status code instead of
failing silently.
NOT TESTED AGAINST LIVE SERVICE: exercised only with httpx.MockTransport in
tests/test_azure_adapters.py.
INTERVIEW ONE-LINER: "The Teams mock writes the exact webhook payload to a
log; going live is swapping the mock's post() for this function — same
arguments, plus a timeout."
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


def post_teams_webhook(url: str, text: str, *,
                       payload: dict[str, Any] | None = None,
                       timeout: float = 10.0,
                       transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    """POST `text` (or a prepared `payload`) to a Teams webhook/Workflows URL.

    The default payload {"text": text} targets the CLASSIC Teams incoming
    webhook. VERSION-SENSITIVE: Workflows connectors want an adaptive-card
    body; pass `payload` prepared for your receiver if so. Returns the
    service's JSON response."""
    if not url:
        raise ValueError("Teams webhook URL must not be empty")
    body = payload if payload is not None else {"text": text}
    client = httpx.Client(transport=transport, timeout=timeout) \
        if transport is not None else httpx.Client(timeout=timeout)
    try:
        response = client.post(url, json=body)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Teams webhook returned HTTP {response.status_code}: "
                f"{response.text[:200]}")
        try:
            return response.json()
        except ValueError:
            return {"status": response.status_code, "body": response.text[:200]}
    finally:
        client.close()
