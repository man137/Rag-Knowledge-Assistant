"""Prompt loading, message building and context token budgeting.

WHAT: load a prompt template by version from prompts/ (v1/v2/v3), wrap each
retrieved chunk in a clearly delimited [source:chunk_id] block, and pack the
context under a token budget, dropping the LOWEST-ranked chunks first.
WHY: prompts are versioned artefacts evaluated like code (rag/evaluation/
prompt_ab.py measures v1 vs v2 vs v3 on the same set); the budget guarantees
the prompt never overflows the model window and keeps cost predictable.
MATHS: budget arithmetic — the prompt must satisfy
    count(instructions) + count(question) + count(context) <= budget
Chunks enter in rank order (best first) and are dropped from the END of the
list (lowest rank) until the sum fits, so the best chunks always survive and
the caller learns exactly which chunks were dropped. Token counts use the
regex tokeniser from rag.cleaning (words, not sub-words) — an approximation
on purpose: no tiktoken dependency, and deliberately conservative budgets.
FAILURE MODE: a template without {context}/{question} placeholders renders
silently wrong, so load_template validates them and raises immediately.
.format() would also choke on stray braces — templates must contain only the
two placeholders.
INTERVIEW ONE-LINER: "Prompts are versioned files A/B-tested on the same
eval set, and the context packer enforces a token budget by dropping the
lowest-ranked chunks first with the ids reported back."
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from rag.cleaning import tokens
from rag.hybrid import RetrievalResult

PROMPT_VERSIONS = ("v1", "v2", "v3")
# The canonical refusal sentence, byte-identical in v2/v3 templates, in the
# pipeline's no-answer path, and in evaluation expectations.
REFUSAL = "I don't know based on the provided documents."

Chunk = Sequence


def load_template(version: str, prompts_dir: str | Path) -> str:
    path = Path(prompts_dir) / f"prompt_{version}.txt"
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown prompt version {version!r}; "
                         f"available: {PROMPT_VERSIONS}")
    try:
        template = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Prompt template {path} not found") from exc
    if "{context}" not in template or "{question}" not in template:
        raise ValueError(
            f"Template {path} must contain {{context}} and {{question}}")
    return template


def wrap_chunk(chunk_id: str, text: str) -> str:
    """One clearly delimited context block so the model can quote the id and
    the generator can validate the citation afterwards."""
    return f"[source:{chunk_id}]\n{text}\n[/source]"


def build_messages(
    question: str,
    results: Sequence[RetrievalResult],
    prompt_version: str,
    prompts_dir: str | Path,
    context_token_budget: int,
    max_output_tokens: int = 512,
) -> tuple[list[dict[str, str]], list[str], int]:
    """Build the message list for the generator.

    Returns (messages, dropped_chunk_ids, prompt_token_count). Chunks arrive
    ranked best-first; the packer drops from the tail (lowest rank) until the
    budget fits, never the other way round.
    """
    template = load_template(prompt_version, prompts_dir)
    base = template.replace("{context}", "")
    overhead = len(tokens(base)) + len(tokens(question)) + 2
    budget = max(0, context_token_budget - overhead)

    kept: list[RetrievalResult] = list(results)
    dropped: list[str] = []
    used = 0
    blocks: list[str] = []
    for res in kept:
        block = wrap_chunk(res.chunk.chunk_id, res.chunk.text)
        cost = len(tokens(block))
        if used + cost > budget and blocks:
            # budget exhausted: everything AFTER this point is dropped
            break
        blocks.append(block)
        used += cost
    # anything not packed is "dropped" (lowest-ranked chunks first, because
    # `results` is ranked best-first and we stop at the first overflow)
    dropped = [res.chunk.chunk_id for res in kept[len(blocks):]]
    context = "\n\n".join(blocks)
    user_content = template.replace("{context}", context)\
                           .replace("{question}", question)
    messages = [{"role": "user", "content": user_content}]
    total = overhead + used
    return messages, dropped, total


def answer_no_answer() -> str:
    """The pipeline's no-answer response — identical to the templates'."""
    return REFUSAL
