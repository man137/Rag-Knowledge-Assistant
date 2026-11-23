"""Tests for the CrmMock and TeamsMock stores."""

from __future__ import annotations

from pathlib import Path

from agent.crm_mock import CrmMock
from agent.teams_mock import TeamsMock


def test_crm_create_get_list_roundtrip(settings):
    crm = CrmMock(settings.crm_db_path)
    try:
        tid = crm.create_ticket("Refund dispute above INR 50,000", "high")
        assert tid == 1
        tid2 = crm.create_ticket("Badge reprints queued", "low")
        assert tid2 == 2
        ticket = crm.get_ticket(tid)
        assert ticket["summary"].startswith("Refund dispute")
        assert ticket["priority"] == "high"
        assert ticket["status"] == "open"
        assert ticket["created_at"]
        assert crm.get_ticket(999) is None
        listing = crm.list_tickets()
        assert [t["id"] for t in listing] == [2, 1]
        assert crm.count() == 2
    finally:
        crm.close()


def test_crm_validates_priority_and_summary(settings):
    import pytest

    crm = CrmMock(settings.crm_db_path)
    try:
        with pytest.raises(ValueError):
            crm.create_ticket("x", priority="urgent")
        with pytest.raises(ValueError):
            crm.create_ticket("   ")
    finally:
        crm.close()


def test_crm_persists_across_connections(settings):
    first = CrmMock(settings.crm_db_path)
    first.create_ticket("persisted ticket", "medium")
    first.close()
    second = CrmMock(settings.crm_db_path)
    try:
        assert second.count() == 1
        assert second.list_tickets()[0]["summary"] == "persisted ticket"
    finally:
        second.close()


def test_teams_mock_records_in_memory_and_file(settings):
    teams = TeamsMock(settings.teams_log_path)
    message = teams.post("#support-escalations", "high ticket needs an owner")
    assert message["channel"] == "#support-escalations"
    assert message["ts"] and message["text"]
    assert teams.all_messages() == [message]
    # the exact payload the real incoming webhook would receive is in the file
    log_lines = Path(settings.teams_log_path).read_text(encoding="utf-8")\
        .strip().splitlines()
    assert len(log_lines) == 1
    assert "#support-escalations" in log_lines[0]


def test_teams_mock_validates_text(settings):
    import pytest

    teams = TeamsMock(settings.teams_log_path)
    with pytest.raises(ValueError):
        teams.post("#general", "  ")
