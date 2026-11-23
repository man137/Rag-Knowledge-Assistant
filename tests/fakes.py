"""One import point for every fake used by the test-suite.

WHAT: re-exports FakeEmbedder (rag.embedder), FakeLLM (rag.generator) and
FakeReranker (rag.reranker), plus helpers: context_citing_llm builds a
FakeLLM that answers with the chunk ids actually present in the prompt, and
scripted_llm wraps a list of answers for the agent loop and the API tests.
WHY: tests must run fully offline, with deterministic outputs, and never
download a model or call a live LLM.
MATHS: none.
FAILURE MODE: a fake that "knows" the corpus would make retrieval tests pass
vacuously; the helpers only read the ids from the rendered prompt, so a valid
citation in a test is earned by retrieval, not hard-coded.
INTERVIEW ONE-LINER: "The suite runs on deterministic fakes wired through the
same Protocols as the real providers, so offline tests exercise the full
pipeline without a model."
"""

from __future__ import annotations

import re

from rag.embedder import FakeEmbedder
from rag.generator import FakeLLM
from rag.reranker import FakeReranker

__all__ = ["FakeEmbedder", "FakeLLM", "FakeReranker",
           "context_citing_llm", "scripted_llm", "first_source_from_messages"]


def first_source_from_messages(messages: list[dict[str, str]]) -> str | None:
    """The first [source:chunk_id] block in the last user message."""
    content = messages[-1]["content"] if messages else ""
    match = re.search(r"\[source:([0-9a-f]{12})\]", content)
    return match.group(1) if match else None


def context_citing_llm(answer_text: str = "Here is the answer from the context.",
                       add_citation: bool = True) -> FakeLLM:
    """A FakeLLM that answers citing the first chunk id found in its prompt.

    Valid citations in tests are therefore earned by real retrieval: the id
    comes from the rendered context, never hard-coded in the test."""

    def fn(messages: list[dict[str, str]]) -> str:
        cid = first_source_from_messages(messages)
        if add_citation and cid:
            return f"{answer_text} [source:{cid}]"
        return answer_text

    return FakeLLM(fn=fn)


def scripted_llm(*answers: str) -> FakeLLM:
    """A FakeLLM popping the given answers one per generate() call."""
    return FakeLLM(responses=list(answers))
