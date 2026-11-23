"""Tests for rag.chunking: limits, exact overlap, no text loss, stable ids."""

from __future__ import annotations

import pytest

from rag.chunking import Chunk, FixedTokenChunker, SentenceAwareChunker, chunk_id_for, make_chunker
from rag.cleaning import TOKEN_RE, clean_text

# 4-token sentences keep the arithmetic in tests easy to reason about.
LONG = " ".join(f"Rule {i} is clear." for i in range(80))


def _covered(text: str, chunks: list[Chunk]) -> bool:
    """Every non-whitespace character of the source text falls inside some
    chunk's [start_char, end_char) range (overlap removed by union)."""
    covered: set[int] = set()
    for c in chunks:
        covered.update(range(c.start_char, c.end_char))
    return all(p in covered for p, ch in enumerate(text) if not ch.isspace())


def test_fixed_chunker_respects_limit_and_exact_overlap():
    text = clean_text(LONG)
    chunks = FixedTokenChunker(chunk_size=20, chunk_overlap=5).chunk(
        text, "doc.md", "Doc", 1, ["all"])
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 20
    # consecutive windows deliberately differ in length at the tail, hence
    # strict=False on the zip
    for a, b in zip(chunks, chunks[1:], strict=False):
        # exact overlap: the last 5 tokens of a ARE the first 5 of b.
        # window i covers token indexes [i, i+size); the next starts at
        # i + (size-overlap), so shared tokens = size - step = overlap.
        toks_a = TOKEN_RE.findall(text[a.start_char:a.end_char])
        toks_b = TOKEN_RE.findall(text[b.start_char:b.end_char])
        assert toks_a[-5:] == toks_b[:5]
    assert _covered(text, chunks)


def test_fixed_chunker_no_token_loss():
    text = clean_text(" ".join(f"word{i}" for i in range(100)))
    chunks = FixedTokenChunker(12, 3).chunk(text, "s", "Doc", 1, ["all"])
    seen: set[str] = set()
    for c in chunks:
        seen.update(c.text.split())
    assert seen == set(text.split())


def test_chunk_ids_stable_across_runs_and_scoped_by_source():
    text = "alpha beta gamma delta epsilon zeta eta theta."
    a = FixedTokenChunker(3, 1).chunk(text, "s1.md", "T", 1, ["all"])
    b = FixedTokenChunker(3, 1).chunk(text, "s1.md", "T", 1, ["all"])
    c = FixedTokenChunker(3, 1).chunk(text, "s2.md", "T", 1, ["all"])
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b]
    assert a[0].chunk_id != c[0].chunk_id
    assert all(len(x.chunk_id) == 12 for x in a)
    assert a[0].chunk_id == chunk_id_for("s1.md", a[0].start_char)


def test_sentence_chunker_limits_and_sentence_overlap():
    text = clean_text(LONG)
    chunks = SentenceAwareChunker(chunk_size=20, chunk_overlap=5).chunk(
        text, "s.md", "Doc", 1, ["all"])
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 20
    for a, b in zip(chunks, chunks[1:], strict=False):
        # overlap region between consecutive chunks: 0 < tokens <= configured
        overlap_tokens = TOKEN_RE.findall(text[b.start_char:a.end_char])
        assert 0 < len(overlap_tokens) <= 5
    assert _covered(text, chunks)


def test_sentence_chunker_hard_splits_oversized_sentence():
    # one "sentence" of 50 tokens with a chunk limit of 10
    text = clean_text(" ".join(["monsterword"] * 49) + " monsterword.")
    chunks = SentenceAwareChunker(chunk_size=10, chunk_overlap=3).chunk(
        text, "s.md", "Doc", 1, ["all"])
    assert len(chunks) >= 5
    assert all(c.token_count <= 10 for c in chunks)
    assert _covered(text, chunks)


def test_sentence_chunker_survives_headings_and_bullets():
    text = clean_text(
        "## 1. Purpose\n\nFirst real sentence here.\n\n"
        "- bullet without punctuation\n\nSecond sentence here.")
    chunks = SentenceAwareChunker(chunk_size=25, chunk_overlap=5).chunk(
        text, "s.md", "Doc", 1, ["all"])
    assert chunks
    assert all(c.token_count <= 25 for c in chunks)
    assert _covered(text, chunks)


def test_empty_text_yields_no_chunks():
    assert FixedTokenChunker(20, 5).chunk("   ", "s", "T", 1, ["all"]) == []
    assert SentenceAwareChunker(20, 5).chunk("", "s", "T", 1, ["all"]) == []


def test_chunk_serialisation_roundtrip():
    text = clean_text("one two three four five six seven eight.")
    chunks = FixedTokenChunker(4, 1).chunk(text, "s.md", "Title", 2, ["staff"])
    rebuilt = [Chunk.from_dict(c.to_dict()) for c in chunks]
    assert rebuilt == chunks
    assert chunks[0].allowed_roles == ["staff"]
    assert chunks[0].source == "s.md"
    assert chunks[0].title == "Title"
    assert chunks[0].page == 2


def test_make_chunker_factory():
    assert isinstance(make_chunker("fixed", 100, 20), FixedTokenChunker)
    assert isinstance(make_chunker("sentence", 100, 20), SentenceAwareChunker)
    with pytest.raises(ValueError):
        make_chunker("word2vec", 100, 20)
