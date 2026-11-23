"""BM25 (Okapi) with a hand-built inverted index.

WHAT: tokeniser (lowercase regex tokens minus a small stopword list), inverted
index (token -> postings of (doc, term frequency)), Okapi BM25 scoring with
k1=1.5, b=0.75, ACL-aware search that ranks only permitted chunks, and JSON
save/load.
WHY: lexical retrieval catches exact terms ("INR 9,000", ticket numbers,
"FORM-12") that embeddings can miss; fused with dense retrieval it is the
hybrid search pattern every serious RAG system uses.
MATHS: Robertson-Sparck-Jones IDF in the Okapi BM25 form:
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
where N = number of chunks and df(t) = chunks containing t. The +1 inside the
log keeps idf strictly positive, so extremely common terms can never
subtract score. Term contribution:
    idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
with tf = in-chunk frequency of t, dl = chunk length in tokens, avgdl = mean
chunk length. k1=1.5 saturates the gain from repetition; b=0.75 scales length
normalisation (b=0 none, b=1 fully proportional).
FAILURE MODE: no stemming: "refund" does not match "refunds", a real recall
gap that the dense half of hybrid retrieval covers. BM25 scores are absolute
and not comparable across queries, so the no-answer threshold for bm25 mode is
calibrated separately in rag/pipeline.py.
INTERVIEW ONE-LINER: "I implemented Okapi BM25 down to the IDF formula and
can explain what k1 and b do; the same trick powers Azure AI Search's BM25
half of hybrid."
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Sequence

from rag.chunking import Chunk
from rag.cleaning import content_tokens

log = logging.getLogger(__name__)


class BM25Index:
    """Inverted-index BM25 over Chunks, with role filtering before ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        # token -> list of (row, term frequency)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.N = 0
        self._idf: dict[str, float] = {}

    def build(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.postings = {}
        self.doc_lens = []
        for row, chunk in enumerate(self.chunks):
            toks = content_tokens(chunk.text)
            self.doc_lens.append(len(toks))
            counts: dict[str, int] = {}
            for tok in toks:
                counts[tok] = counts.get(tok, 0) + 1
            for tok, tf in counts.items():
                self.postings.setdefault(tok, []).append((row, tf))
        self.N = len(self.chunks)
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N else 0.0
        self._idf = {}
        log.info("BM25 built: %d chunks, %d unique tokens", self.N, len(self.postings))

    def idf(self, token: str) -> float:
        """Cached idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))."""
        cached = self._idf.get(token)
        if cached is not None:
            return cached
        df = len(self.postings.get(token, ()))
        value = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
        self._idf[token] = value
        return value

    def _score_chunk(self, query_tokens: Sequence[str], row: int) -> float:
        dl = self.doc_lens[row]
        denom_norm = self.k1 * (1.0 - self.b + self.b * dl / self.avgdl) \
            if self.avgdl > 0 else self.k1
        score = 0.0
        seen: set[str] = set()
        for tok in query_tokens:
            if tok in seen:
                continue  # each query term contributes once
            seen.add(tok)
            tf = 0
            for r, f in self.postings.get(tok, ()):
                if r == row:
                    tf = f
                    break
            if tf == 0:
                continue
            # Okapi BM25 term contribution (see module docstring)
            score += self.idf(tok) * tf * (self.k1 + 1.0) / (tf + denom_norm)
        return score

    def search(self, query: str, k: int,
               allowed_roles: set[str] | None = None) -> list[tuple[Chunk, float]]:
        """Top-k chunks for the query, ranking only ACL-permitted rows."""
        if self.N == 0 or k <= 0:
            return []
        q_tokens = content_tokens(query)
        if not q_tokens:
            return []
        permitted: set[int] | None = None
        if allowed_roles is not None:
            permitted = {i for i, c in enumerate(self.chunks)
                         if allowed_roles.intersection(c.allowed_roles)}
            if not permitted:
                return []
        # postings-driven scoring: only chunks containing query terms are touched
        candidates: set[int] = set()
        for tok in set(q_tokens):
            for row, _tf in self.postings.get(tok, ()):
                if permitted is None or row in permitted:
                    candidates.add(row)
        scored: list[tuple[float, int]] = []
        for row in candidates:
            scored.append((self._score_chunk(q_tokens, row), row))
        # deterministic order: score desc, then row index
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(self.chunks[row], score) for score, row in scored[:k]]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1, "b": self.b, "avgdl": self.avgdl, "N": self.N,
            "doc_lens": self.doc_lens,
            "postings": {tok: [[row, tf] for row, tf in rows]
                         for tok, rows in self.postings.items()},
            "chunks": [c.to_dict() for c in self.chunks],
        }
        p.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        index = cls(k1=payload["k1"], b=payload["b"])
        index.avgdl = float(payload["avgdl"])
        index.N = int(payload["N"])
        index.doc_lens = [int(x) for x in payload["doc_lens"]]
        index.postings = {tok: [(int(row), int(tf)) for row, tf in rows]
                          for tok, rows in payload["postings"].items()}
        index.chunks = [Chunk.from_dict(d) for d in payload["chunks"]]
        return index
