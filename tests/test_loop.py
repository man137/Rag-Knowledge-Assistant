"""Tests for agent.loop: success, errors, repair, step limit, safety gates."""

from __future__ import annotations

import json

import pytest

from agent.crm_mock import CrmMock
from agent.loop import AgentLoop
from agent.teams_mock import TeamsMock
from agent.tools import build_default_registry
from rag.embedder import FakeEmbedder
from rag.generator import FakeLLM
from rag.pipeline import chunk_corpus, pipeline_from_chunks
from tests.fakes import context_citing_llm

GOAL = "A customer asks about the refund rule for an event cancellation."


def _action(action, **args):
    return json.dumps({"thought": "planned", "action": action, "args": args})


def _final(answer):
    return json.dumps({"thought": "done", "action": "final_answer",
                       "answer": answer})


@pytest.fixture()
def stack(settings, docs):
    chunks = chunk_corpus(settings, docs)
    pipeline = pipeline_from_chunks(settings, chunks, FakeEmbedder(),
                                    context_citing_llm())
    crm = CrmMock(settings.crm_db_path)
    teams = TeamsMock(settings.teams_log_path)
    registry = build_default_registry(pipeline, crm, teams, user_role="guest")
    yield registry, crm, teams
    crm.close()


def _loop(registry, llm, max_steps=8):
    return AgentLoop(registry, llm, max_steps=max_steps)


def test_success_path_with_tool_then_final_answer(stack):
    registry, _, _ = stack
    llm = FakeLLM(responses=[
        _action("search_kb", query="refund rule event cancellation"),
        _final("You get a 50 percent refund 15 to 29 days before the event. "
               "[source:abc123def456]"),
    ])
    result = _loop(registry, llm).run(GOAL)
    assert result.ok is True
    assert result.steps_used == 2
    assert "50 percent" in result.final_answer
    assert len(result.trace) == 2
    assert result.trace[0]["action"] == "search_kb"
    assert result.trace[0]["observation"]["ok"] is True


def test_tool_error_is_fed_back_as_observation(stack):
    registry, crm, _ = stack
    llm = FakeLLM(responses=[
        _action("lookup_ticket", id=999999),          # ticket does not exist
        _final("I could not find that ticket."),
    ])
    result = _loop(registry, llm).run("Find ticket 999999")
    assert result.ok is True
    assert result.trace[0]["ok"] is False
    assert "no ticket with id 999999" in \
        json.dumps(result.trace[0]["observation"])
    # the loop still recovered and finished in 2 steps
    assert result.steps_used == 2


def test_malformed_json_gets_exactly_one_repair_attempt(stack):
    registry, _, _ = stack
    llm = FakeLLM(responses=[
        "I will search the refund rules now (not JSON)",
        _final("Repaired answer with citation [source:000000000000]."),
    ])
    result = _loop(registry, llm).run(GOAL)
    assert result.ok is True
    assert any("invalid JSON" in json.dumps(entry) for entry in result.trace)
    assert result.steps_used == 2


def test_two_bad_replies_stop_the_loop(stack):
    registry, _, _ = stack
    llm = FakeLLM(responses=["no json here", "still not json"])
    result = _loop(registry, llm).run(GOAL)
    assert result.ok is False
    assert "valid action plan" in result.final_answer
    assert any("after repair attempt" in json.dumps(e) for e in result.trace)


def test_step_limit_terminates_loop(stack):
    registry, _, _ = stack
    infinite = FakeLLM(fn=lambda messages: _action("search_kb", query="again"))
    result = _loop(registry, infinite, max_steps=3).run(GOAL)
    assert result.ok is False
    assert result.steps_used == 3
    assert "3 steps" in result.final_answer


def test_non_allowlisted_tool_is_refused_and_recovered(stack):
    registry, _, _ = stack
    llm = FakeLLM(responses=[
        _action("delete_everything", path="/"),
        _final("I cannot delete anything; I only answer policy questions."),
    ])
    result = _loop(registry, llm).run(GOAL)
    assert result.ok is True
    assert "not on the allowlist" in \
        json.dumps(result.trace[0]["observation"])
    assert result.trace[0]["ok"] is False


def test_write_without_confirm_refused_then_confirmed_writes(stack):
    registry, crm, teams = stack
    llm = FakeLLM(responses=[
        _action("create_crm_ticket", summary="refund dispute",
                priority="high"),                       # confirm missing
        _action("create_crm_ticket", summary="refund dispute",
                priority="high", confirm=True),
        _action("post_teams_message", channel="#support-escalations",
                text="escalating ticket", confirm=True),
        _final("Escalated: ticket created and team notified."),
    ])
    result = _loop(registry, llm).run(GOAL)
    assert result.ok is True
    assert crm.count() == 1, "exactly one ticket must have been created"
    assert "confirm=true" in json.dumps(result.trace[0]["observation"])
    assert teams.messages[0]["channel"] == "#support-escalations"


def test_trace_records_timestamps_for_every_step(stack):
    registry, _, _ = stack
    llm = FakeLLM(responses=[_final("Immediate answer.")])
    result = _loop(registry, llm).run(GOAL)
    assert all("ts" in entry for entry in result.trace)
    assert result.steps_used == 1
