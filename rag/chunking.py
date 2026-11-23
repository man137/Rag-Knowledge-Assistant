"""Chunking: split cleaned document text into overlapping Chunks.

WHAT: one `Chunk` dataclass and two strategies behind a common interface —
FixedTokenChunker (sliding token window, exact overlap) and
SentenceAwareChunker (pack whole sentences up to the token limit, overlap by
whole trailing sentences). chunk_id is the first 12 hex chars of
sha1(source + ":" + start_char) so IDs are stable across runs.
WHY: chunk size is the single most leveraged retrieval parameter; overlap
prevents an answer from being cut in half at a boundary; sentence boundaries
keep each chunk semantically coherent.
MATHS: overlap. A chunk holds `size` tokens; the next chunk starts `step =
size - overlap` tokens later, so consecutive windows share exactly `overlap`
tokens (for the fixed strategy). Reconstructing the text from the union of
chunk char-ranges loses nothing because the windows tile every token span.
Token counts use the regex tokeniser from rag.cleaning when the embedding
model's tokenizer is unavailable (see FixedTokenChunker.tokenizer) — the two
differ slightly, so counts are approximate; tests only rely on the regex one.
FAILURE MODE: a sentence longer than `chunk_size` would break the invariant
"chunk <= limit", so SentenceAwareChunker hard-splits such sentences on token
boundaries; headings with numbers ("1. Purpose") yield tiny sentence
fragments, which is harmless for packing but slightly odd to read.
INTERVIEW ONE-LINER: "I built both a fixed sliding window with exact overlap
and a sentence-aware packer, then measured which one wins on the eval set."
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from rag.cleaning import TOKEN_RE

log = logging.getLogger(__name__)

_SENT_TERM_RE = re.compile(r"[.!?]+")


def chunk_id_for(source: str, start_char: int, page: int = 1) -> str:
    """Stable chunk id: first 12 hex chars of sha1(source + ':' + start_char).
    Stable across runs, unique per (document, page, position)."""
    key = f"{source}:p{page}:{start_char}" if page > 1 else f"{source}:{start_char}"
    digest = hashlib.sha1(key.encode("utf-8"))
    return digest.hexdigest()[:12]


@dataclass
class Chunk:
    """One retrievable unit of text with its provenance and ACL."""

    chunk_id: str
    text: str
    source: str
    title: str
    page: int
    start_char: int
    end_char: int
    token_count: int
    allowed_roles: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "text": self.text, "source": self.source,
            "title": self.title, "page": self.page, "start_char": self.start_char,
            "end_char": self.end_char, "token_count": self.token_count,
            "allowed_roles": list(self.allowed_roles),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(chunk_id=d["chunk_id"], text=d["text"], source=d["source"],
                   title=d["title"], page=int(d["page"]),
                   start_char=int(d["start_char"]), end_char=int(d["end_char"]),
                   token_count=int(d["token_count"]),
                   allowed_roles=list(d["allowed_roles"]))


class Chunker(Protocol):
    """Common interface: split one document's cleaned text into Chunks."""

    def chunk(self, text: str, source: str, title: str, page: int,
              allowed_roles: Sequence[str]) -> list[Chunk]: ...


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _grow_left(text: str, pos: int) -> int:
    """Extend a chunk start left over adjacent non-whitespace (punctuation
    clinging to the first token) so nothing is lost between chunks."""
    while pos > 0 and not text[pos - 1].isspace():
        pos -= 1
    return pos


def _grow_right(text: str, pos: int) -> int:
    while pos < len(text) and not text[pos].isspace():
        pos += 1
    return pos


class FixedTokenChunker:
    """Sliding window over regex tokens: `chunk_size` tokens per chunk, exactly
    `chunk_overlap` tokens shared with the next chunk.

    `tokenizer` may be the embedding model's real tokenizer; when None (model
    not installed) we fall back to the regex tokeniser from rag.cleaning and
    SAY SO HERE: regex tokens approximate sub-word tokens, so a "500-token"
    chunk is closer to 500 words — good enough for budget arithmetic, and the
    same measure is used by the token-budget logic in rag.prompt."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80,
                 tokenizer: Any | None = None) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer  # optional: object with .encode() -> list

    def _count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(TOKEN_RE.findall(text))

    def chunk(self, text: str, source: str, title: str, page: int,
              allowed_roles: Sequence[str]) -> list[Chunk]:
        spans = _token_spans(text)
        if not spans:
            return []
        size = self.chunk_size
        step = size - self.chunk_overlap
        chunks: list[Chunk] = []
        i = 0
        while True:
            window = spans[i:i + size]
            start = _grow_left(text, window[0][0])
            end = _grow_right(text, window[-1][1])
            piece = text[start:end]
            if piece.strip():
                # exact overlap invariant: window i covers token indexes
                # [i, i+size); window i+step covers [i+step, ...) — shared
                # tokens = size - step = chunk_overlap.
                chunks.append(Chunk(
                    chunk_id=chunk_id_for(source, start, page), text=piece,
                    source=source, title=title, page=page,
                    start_char=start, end_char=end, token_count=len(window),
                    allowed_roles=list(allowed_roles)))
            if i + size >= len(spans):
                break
            i += step
        return chunks


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Char spans of sentences: scan to the next sentence terminator (.!?),
    but stop early at a blank line so unpunctuated headings/bullets become
    their own "sentence" instead of swallowing the rest of the document."""
    spans: list[tuple[int, int]] = []
    pos, n = 0, len(text)
    while pos < n:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        m = _SENT_TERM_RE.search(text, pos)
        para_end = text.find("\n\n", pos)
        if m and (para_end == -1 or m.start() < para_end):
            end = m.end()
        else:
            end = para_end if para_end != -1 else n
        if end <= pos:  # safety: never emit an empty span
            end = min(n, pos + 1)
        spans.append((pos, end))
        pos = end
    return spans


class SentenceAwareChunker:
    """Pack whole sentences until the token limit; start the next chunk with
    the trailing sentences (up to `chunk_overlap` tokens) of the previous one,
    so overlap never cuts a sentence in half. A single sentence longer than
    the limit is hard-split on token boundaries to preserve the
    'no chunk exceeds chunk_size' invariant."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80,
                 tokenizer: Any | None = None) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer

    def _count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(TOKEN_RE.findall(text))

    def chunk(self, text: str, source: str, title: str, page: int,
              allowed_roles: Sequence[str]) -> list[Chunk]:
        spans = sentence_spans(text)
        if not spans:
            return []
        chunks: list[Chunk] = []
        cur: list[tuple[int, int]] = []
        cur_tokens = 0

        def emit(window: list[tuple[int, int]]) -> None:
            start = _grow_left(text, window[0][0])
            end = _grow_right(text, window[-1][1])
            piece = text[start:end]
            chunks.append(Chunk(
                chunk_id=chunk_id_for(source, start, page), text=piece,
                source=source, title=title, page=page, start_char=start,
                end_char=end, token_count=self._count(piece),
                allowed_roles=list(allowed_roles)))

        def carry_overlap(window: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int]:
            kept: list[tuple[int, int]] = []
            n = 0
            for s in reversed(window):
                st = self._count(text[s[0]:s[1]])
                if n + st > self.chunk_overlap:
                    break
                kept.insert(0, s)
                n += st
            return kept, n

        for span in spans:
            st = self._count(text[span[0]:span[1]])
            if st > self.chunk_size:
                # pathological sentence: flush current buffer, then hard-split
                if cur:
                    emit(cur)
                    cur, cur_tokens = [], 0
                pieces = self._hard_split(text, span)
                for piece in pieces:
                    emit([piece])
                continue
            if cur and cur_tokens + st > self.chunk_size:
                emit(cur)
                cur, cur_tokens = carry_overlap(cur)
            cur.append(span)
            cur_tokens += st
        if cur:
            emit(cur)
        return chunks

    def _hard_split(self, text: str,
                    span: tuple[int, int]) -> list[tuple[int, int]]:
        """Token-boundary split of one over-long sentence into spans, each
        within the token limit, with no overlap (degenerate case)."""
        out: list[tuple[int, int]] = []
        spans = _token_spans(text[span[0]:span[1]])
        step = max(1, self.chunk_size - self.chunk_overlap)
        for j in range(0, len(spans), step):
            window = spans[j:j + self.chunk_size]
            out.append((span[0] + window[0][0], span[0] + window[-1][1]))
            if j + self.chunk_size >= len(spans):
                break
        return out


def make_chunker(strategy: str, chunk_size: int, chunk_overlap: int,
                  tokenizer: Any | None = None) -> Chunker:
    """Factory by strategy name ("fixed" | "sentence")."""
    if strategy == "fixed":
        return FixedTokenChunker(chunk_size, chunk_overlap, tokenizer)
    if strategy == "sentence":
        return SentenceAwareChunker(chunk_size, chunk_overlap, tokenizer)
    raise ValueError(f"unknown chunking strategy {strategy!r}")
