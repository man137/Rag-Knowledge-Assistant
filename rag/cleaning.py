"""Text cleaning for corpus documents.

WHAT: unicode normalisation (NFKC), whitespace collapsing, paragraph-break
preservation, and a heuristic that strips repeated header/footer lines
(identical lines appearing as the first or last line of many blocks).
Also home of the shared tokeniser used by BM25, FakeEmbedder and the
chunkers (regex word tokens, lowercase, with a small stopword list).
WHY: garbage in, garbage out — repeated page headers poison embeddings and
split answers across chunks; consistent tokenisation keeps BM25 and the fake
embedder comparable in tests.
MATHS: none.
FAILURE MODE: the header/footer heuristic keys on line POSITION and exact
match, so it misses headers with page numbers ("Page 3 of 12") or slight
formatting changes, and could in principle strip a legitimately repeated line.
It is a heuristic; evaluation catches the damage.
INTERVIEW ONE-LINER: "I strip repeated page headers at ingestion because they
pollute embeddings and produce junk chunks at fixed-size boundaries."
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import replace

from rag.loaders import Document

log = logging.getLogger(__name__)

# Word tokens: letters/digits with internal hyphens/apostrophes ("INR 2,500"
# -> "inr", "2", "500"; "state-of-the-art" -> "state-of-the-art").
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")

# Small stopword list (kept tiny so BM25 stays explainable in interviews).
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "his",
    "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our",
    "she", "that", "the", "their", "them", "they", "this", "to", "us", "was",
    "we", "were", "will", "with", "would", "you", "your",
})


def tokens(text: str) -> list[str]:
    """Lowercased word tokens, stopwords INCLUDED (used for token counts)."""
    return TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Lowercased word tokens minus stopwords (used by BM25 and FakeEmbedder,
    where function words carry no signal)."""
    return [t for t in tokens(text) if t not in STOPWORDS]


def normalise_text(text: str) -> str:
    """NFKC-normalise and tidy whitespace while keeping paragraph breaks
    (a blank line between blocks)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)        # collapse runs of spaces/tabs
    text = re.sub(r" +\n", "\n", text)         # strip trailing spaces per line
    text = re.sub(r"\n{3,}", "\n\n", text)     # >2 newlines = one blank line
    return text.strip()


def strip_repeated_lines(text: str, min_blocks: int = 3) -> str:
    """Drop lines that act as headers/footers: a line appearing as the FIRST
    or LAST non-empty line of at least `min_blocks` blocks is treated as page
    furniture and removed. Blocks are paragraphs separated by blank lines."""
    blocks = text.split("\n\n")
    indexed = [b for b in blocks if b.strip()]
    if len(indexed) < min_blocks:
        return text
    first_seen: dict[str, set[int]] = {}
    last_seen: dict[str, set[int]] = {}
    for i, block in enumerate(indexed):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        first_seen.setdefault(lines[0], set()).add(i)
        last_seen.setdefault(lines[-1], set()).add(i)
    drop = {ln for ln, seen in first_seen.items() if len(seen) >= min_blocks}
    drop |= {ln for ln, seen in last_seen.items() if len(seen) >= min_blocks}
    if not drop:
        return text
    kept_blocks: list[str] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() not in drop]
        piece = "\n".join(lines).strip()
        if piece:
            kept_blocks.append(piece)
    return "\n\n".join(kept_blocks)


def clean_text(text: str) -> str:
    """Full cleaning pipeline for one document's raw text."""
    return strip_repeated_lines(normalise_text(text))


def clean_document(doc: Document) -> Document:
    """Return a copy of `doc` with cleaned text (metadata untouched)."""
    cleaned = clean_text(doc.text)
    if cleaned != doc.text:
        log.debug("Cleaned document %s (page %s)", doc.source, doc.page)
    return replace(doc, text=cleaned)
