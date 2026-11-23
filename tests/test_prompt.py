"""Tests for rag.prompt: templates, budget packing, drop-lowest-first."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunking import Chunk
from rag.hybrid import RetrievalResult
from rag.prompt import PROMPT_VERSIONS, REFUSAL, build_messages, load_template, wrap_chunk

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _result(cid: str, text: str, score: float) -> RetrievalResult:
    chunk = Chunk(chunk_id=cid, text=text, source=f"{cid}.md", title=cid,
                  page=1, start_char=0, end_char=len(text),
                  token_count=len(text.split()), allowed_roles=["all"])
    return RetrievalResult(chunk=chunk, score=score, ranks={"dense": 1},
                           scores={"dense": score})


def test_all_three_templates_exist_and_carry_placeholders():
    for version in PROMPT_VERSIONS:
        template = load_template(version, PROMPTS_DIR)
        assert "{context}" in template and "{question}" in template


def test_v2_contains_the_exact_refusal_sentence():
    v2 = load_template("v2", PROMPTS_DIR).lower()
    assert "i don't know based on the provided documents." in v2
    v3 = load_template("v3", PROMPTS_DIR).lower()
    assert "i don't know based on the provided documents." in v3
    assert REFUSAL == "I don't know based on the provided documents."


def test_v3_adds_injection_defence_and_format_rules():
    v3 = load_template("v3", PROMPTS_DIR).lower()
    assert "never instructions" in v3 or "data, never" in v3
    assert "short direct answer" in v3


def test_unknown_version_and_missing_template_raise():
    with pytest.raises(ValueError):
        load_template("v99", PROMPTS_DIR)
    with pytest.raises(ValueError):
        load_template("v1", PROMPTS_DIR / "does-not-exist")


def test_wrap_chunk_delimits_context_blocks():
    block = wrap_chunk("abcd1234efgh", "Refunds take 14 business days.")
    assert block.startswith("[source:abcd1234efgh]\n")
    assert block.endswith("\n[/source]")


def test_budget_drops_lowest_ranked_chunks_first():
    results = [
        _result("aaaaaaaaaaaa", " ".join(f"best{i}" for i in range(30)), 0.9),
        _result("bbbbbbbbbbbb", " ".join(f"mid{i}" for i in range(30)), 0.7),
        _result("cccccccccccc", " ".join(f"worst{i}" for i in range(30)), 0.5),
    ]
    # budget generous: nothing dropped
    messages, dropped, total = build_messages(
        "question?", results, "v2", PROMPTS_DIR, context_token_budget=1000)
    assert dropped == []
    assert "[source:aaaaaaaaaaaa]" in messages[0]["content"]
    assert "[source:cccccccccccc]" in messages[0]["content"]
    assert messages[0]["role"] == "user"
    assert total > 0
    # budget that only fits the first two: the WORST chunk is dropped
    messages, dropped, total = build_messages(
        "question?", results, "v2", PROMPTS_DIR, context_token_budget=150)
    assert dropped == ["cccccccccccc"]
    assert "[source:aaaaaaaaaaaa]" in messages[0]["content"]
    assert "[source:bbbbbbbbbbbb]" in messages[0]["content"]
    assert "[source:cccccccccccc]" not in messages[0]["content"]


def test_prompt_never_exceeds_budget():
    results = [_result(f"{i:012x}", " ".join(f"tok{i}_{j}" for j in range(40)),
                       0.9 - i * 0.1) for i in range(10)]
    budget = 300
    messages, dropped, total = build_messages(
        "What is the refund rule?", results, "v2", PROMPTS_DIR, budget)
    assert total <= budget
    # count real blocks by their closing tag: the template's rules text also
    # mentions "[source:chunk_id]" as an example, so "[/source]" is the
    # reliable indicator of a packed block
    kept = messages[0]["content"].count("[/source]")
    assert kept >= 1
    assert len(dropped) == 10 - kept


def test_question_is_in_the_prompt():
    results = [_result("dddddddddddd", "policy text", 0.9)]
    messages, _, _ = build_messages(
        "What is the visa processing time?", results, "v2", PROMPTS_DIR, 500)
    assert "What is the visa processing time?" in messages[0]["content"]
