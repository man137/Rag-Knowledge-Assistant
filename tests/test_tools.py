"""Tests for agent.tools: allowlist, schema validation, confirm gating."""

from __future__ import annotations

import pytest

from agent.crm_mock import CrmMock
from agent.teams_mock import TeamsMock
from agent.tools import ALLOWED_TOOLS, ToolRegistry, build_default_registry
from rag.embedder import FakeEmbedder
from rag.pipeline import chunk_corpus, pipeline_from_chunks
from tests.fakes import context_citing_llm

REFUND = "What is the refund rule for event cancellation?"


@pytest.fixture()
def crm(settings):
    store = CrmMock(settings.crm_db_path)
    yield store
    store.close()


@pytest.fixture()
def teams(settings):
    return TeamsMock(settings.teams_log_path)


@pytest.fixture()
def registry(settings, docs, crm, teams):
    chunks = chunk_corpus(settings, docs)
    pipeline = pipeline_from_chunks(settings, chunks, FakeEmbedder(),
                                    context_citing_llm())
    return build_default_registry(pipeline, crm, teams, user_role="guest")


def test_allowlist_contains_exactly_the_four_tools():
    assert ALLOWED_TOOLS == ("search_kb", "lookup_ticket", "create_crm_ticket",
                            "post_teams_message")


def test_non_allowlisted_tool_is_never_executed(registry):
    for bad_name in ("rm_rf", "delete_everything", "eval", "search_kb2"):
        result = registry.call(bad_name, {})
        assert result["ok"] is False
        assert "not on the allowlist" in result["error"]


def test_search_kb_returns_grounded_answer(registry):
    result = registry.call("search_kb", {"query": REFUND})
    assert result["ok"] is True
    payload = result["result"]
    assert payload["no_answer"] is False
    assert payload["answer"]                       # grounded by the fake LLM
    assert payload["sources"]
    cited = payload["sources"][0]["chunk_id"]
    assert cited in payload["citations"]           # citations reference the
                                                   # chunks actually retrieved
    assert result["result"]["sources"][0]["source"].endswith(
        "event_registration_faq.md")


def test_search_kb_out_of_scope_returns_no_answer(registry):
    result = registry.call("search_kb", {"query": "colourless green ideas"})
    assert result["ok"] is True
    assert result["result"]["no_answer"] is True


def test_schema_validation_rejects_bad_arguments(registry):
    cases = [
        ("search_kb", {}, "missing required argument"),
        ("search_kb", {"query": 123}, "must be a string"),
        ("search_kb", {"query": ""}, "must not be empty"),
        ("search_kb", {"query": "x", "extra": 1}, "unknown argument"),
        ("create_crm_ticket", {"summary": "s", "priority": "urgent",
                               "confirm": True}, "must be one of"),
        ("create_crm_ticket", {"summary": "s", "priority": "high",
                                "confirm": "yes"}, "confirm=true"),
        ("lookup_ticket", {"id": "7"}, "must be an integer"),
    ]
    for name, args, expected in cases:
        result = registry.call(name, args)
        assert result["ok"] is False, name
        assert expected in result["error"], (name, result["error"])


def test_write_tool_requires_confirm(registry):
    result = registry.call("create_crm_ticket",
                           {"summary": "refund dispute", "priority": "low"})
    assert result["ok"] is False
    assert "confirm=true" in result["error"]


def test_confirmed_write_creates_ticket_and_post(registry, crm, teams):
    result = registry.call("create_crm_ticket",
                           {"summary": "refund dispute", "priority": "medium",
                            "confirm": True})
    assert result["ok"] is True
    ticket_id = result["result"]
    assert crm.count() == 1
    assert registry.call("lookup_ticket", {"id": ticket_id})["ok"] is True
    posted = registry.call("post_teams_message",
                           {"channel": "#support-escalations",
                            "text": "new escalation ticket",
                            "confirm": True})
    assert posted["ok"] is True
    assert teams.messages[0]["channel"] == "#support-escalations"


def test_lookup_missing_ticket_is_an_obsation_not_a_crash(registry):
    result = registry.call("lookup_ticket", {"id": 999999})
    assert result["ok"] is False
    assert "no ticket with id 999999" in result["error"]


def test_duplicate_registration_rejected():
    registry = ToolRegistry()
    registry.register("echo", lambda x: x,
                      {"properties": {"x": {"type": "string"}},
                       "required": ["x"], "write": False})
    with pytest.raises(ValueError):
        registry.register("echo", lambda x: x)


def test_custom_tool_outside_default_allowlist_never_allowed():
    """Even a REGISTERED tool the model invents args for is validated; but an
    UNREGISTERED name is always refused — the allowlist is the registry."""
    registry = ToolRegistry()
    assert registry.call("search_kb", {"query": "x"})["ok"] is False
