"""Vector store: brute force with ACL-before-ranking, plus an optional IVF.

WHAT: `VectorStore` keeps a float32 matrix and a parallel list of Chunks;
`search` is a dot product on L2-normalised vectors with top-k via
argpartition, and role-based access control applied BEFORE ranking. Save/load
with numpy + JSON. `IVFIndex` is a clearly separated, optional approximate
index: hand-written k-means centroids with inverted lists and an nprobe
parameter, used by scripts/benchmark.py to trade recall for speed.
WHY: no external vector DB — the store is a few hundred lines of NumPy the
student can defend line by line, while mirroring exactly what a managed vector
index does (embed, store, ACL-filter, rank).
MATHS: scores = V @ q with V and q L2-normalised, so the dot product IS the
cosine similarity and larger is better. Top-k selection: np.argpartition(-s,
k-1) moves the k largest scores to the front in O(n) without sorting, then a
full sort of just those k gives the final order. ACCESS CONTROL BEFORE
RANKING: scores of rows whose chunk roles are not permitted are set to -inf
BEFORE argpartition, so a restricted chunk can never occupy a top-k slot and
permitted rows rank among themselves.
IVF: seeded k-means (Lloyd iterations) partitions rows into nlist clusters;
a query compares against all centroids, then scores only rows in the nprobe
nearest clusters' inverted lists — sub-linear instead of O(n) per query.
FAILURE MODE: forgetting to pass allowed_roles means NO filtering (None is
the "no ACL" contract used only by offline tooling); the pipeline always
passes the role allow-set from rag.security. IVF with small nprobe misses
recall (measured by benchmark.py); k-means on tiny corpora degenerates, so
build falls back to fewer lists.
INTERVIEW ONE-LINER: "The ACL is a -inf score mask applied before
argpartition — restricted documents cannot even become candidates, which is
stronger than filtering results afterwards."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from rag.chunking import Chunk

log = logging.getLogger(__name__)


class VectorStore:
    """Brute-force cosine (dot) search with chunk metadata in parallel."""

    def __init__(self, dim: int | None = None) -> None:
        self.matrix: np.ndarray = np.zeros((0, dim or 0), dtype=np.float32)
        self.chunks: list[Chunk] = []

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.size else 0

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        vecs = np.asarray(vectors, dtype=np.float32)
        if len(chunks) != vecs.shape[0]:
            raise ValueError(
                f"{len(chunks)} chunks but {vecs.shape[0]} vectors")
        if len(chunks) and self.matrix.size and vecs.shape[1] != self.matrix.shape[1]:
            raise ValueError(
                f"dimension mismatch: store {self.matrix.shape[1]} vs {vecs.shape[1]}")
        if len(self.matrix) == 0 or self.matrix.size == 0:
            self.matrix = vecs.reshape(len(chunks), -1)
        else:
            self.matrix = np.vstack([self.matrix, vecs])
        self.chunks.extend(chunks)

    def _allowed_mask(self, allowed_roles: set[str] | None) -> np.ndarray:
        """Boolean mask over rows. None = no ACL (offline tooling only).
        A chunk is visible iff its roles intersect the permitted set —
        public chunks carry 'all' and the permitted set always contains
        'all' (see rag.security.effective_allowed_roles)."""
        if allowed_roles is None:
            return np.ones(len(self.chunks), dtype=bool)
        mask = np.zeros(len(self.chunks), dtype=bool)
        for i, chunk in enumerate(self.chunks):
            if allowed_roles.intersection(chunk.allowed_roles):
                mask[i] = True
        return mask

    def search(self, query_vec: np.ndarray, k: int,
               allowed_roles: set[str] | None = None) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity, ACL applied before ranking."""
        if len(self.chunks) == 0 or k <= 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"query dim {q.shape[0]} != store dim {self.matrix.shape[1]}")
        norm = float(np.linalg.norm(q))
        q = q / norm if norm > 0 else q  # defensive: dot == cosine
        scores = self.matrix @ q
        mask = self._allowed_mask(allowed_roles)
        # ACL BEFORE RANKING: forbidden rows can never win a top-k slot.
        scores = np.where(mask, scores, -np.inf)
        n_allowed = int(mask.sum())
        if n_allowed == 0:
            return []
        kk = min(k, n_allowed)
        top = np.argpartition(-scores, kk - 1)[:kk]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.chunks[int(i)], float(scores[int(i)]))
                for i in top if not np.isneginf(scores[int(i)])]

    def delete(self, source: str) -> int:
        """Remove all chunks of one source document; returns rows removed."""
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.matrix = self.matrix[keep]
            self.chunks = [self.chunks[i] for i in keep]
        return removed

    def save(self, dirpath: str | Path) -> None:
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.matrix)
        (d / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks]), encoding="utf-8")

    @classmethod
    def load(cls, dirpath: str | Path) -> "VectorStore":
        d = Path(dirpath)
        matrix = np.load(d / "vectors.npy").astype(np.float32)
        chunks = [Chunk.from_dict(rec)
                  for rec in json.loads((d / "chunks.json").read_text(encoding="utf-8"))]
        store = cls(dim=matrix.shape[1] if matrix.size else 0)
        store.matrix = matrix
        store.chunks = chunks
        return store


class IVFIndex:
    """Optional approximate index: hand-written k-means + inverted lists.

    Clearly separated from VectorStore: used only by scripts/benchmark.py to
    measure recall@10 and latency against the brute-force store. Not wired
    into the serving path (the corpus is small; brute force is honest and
    simple), but the maths is the same one used by real ANN indexes.
    """

    def __init__(self, nlist: int = 16, nprobe: int = 4, seed: int = 42,
                 max_iter: int = 50, tol: float = 1e-6) -> None:
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.seed = int(seed)
        self.max_iter = max_iter
        self.tol = tol
        self.centroids: np.ndarray | None = None
        self.inverted: list[list[int]] = []
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def _kmeans(self, x: np.ndarray, k: int) -> np.ndarray:
        """Lloyd's algorithm on L2-normalised rows; similarity = dot.
        Empty clusters are re-seeded with a random data point (seeded RNG),
        so a build is reproducible for a fixed seed."""
        rng = np.random.default_rng(self.seed)
        n = x.shape[0]
        k = max(1, min(k, n))
        centroids = x[rng.choice(n, size=k, replace=False)].astype(np.float64)
        for _ in range(self.max_iter):
            sims = x @ centroids.T                     # (n, k) cosine sims
            assign = np.argmax(sims, axis=1)
            new_centroids = centroids.copy()
            for c in range(k):
                members = x[assign == c]
                if len(members) == 0:
                    new_centroids[c] = x[rng.integers(n)]  # re-seed empty cluster
                else:
                    mean = members.mean(axis=0)
                    new_centroids[c] = mean / (np.linalg.norm(mean) + 1e-12)
            shift = float(np.max(np.abs(new_centroids - centroids)))
            centroids = new_centroids
            if shift < self.tol:
                break
        return centroids

    def build(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        self.matrix = np.asarray(vectors, dtype=np.float32)
        self.chunks = list(chunks)
        n = self.matrix.shape[0]
        if n == 0:
            log.warning("IVF: empty corpus, nothing to index")
            self.centroids = np.zeros((0, 0))
            self.inverted = []
            return
        nlist = max(1, min(self.nlist, n))
        if nlist != self.nlist:
            log.info("IVF: reducing nlist %d -> %d (corpus too small)",
                     self.nlist, nlist)
        self.centroids = self._kmeans(self.matrix.astype(np.float64), nlist)
        sims = self.matrix @ self.centroids.T
        assign = np.argmax(sims, axis=1)
        self.inverted = [[] for _ in range(self.centroids.shape[0])]
        for row, c in enumerate(assign):
            self.inverted[int(c)].append(row)

    def search(self, query_vec: np.ndarray, k: int,
               allowed_roles: set[str] | None = None,
               nprobe: int | None = None) -> list[tuple[Chunk, float]]:
        """Score only rows in the nprobe nearest clusters, then ACL-mask and
        top-k exactly like VectorStore.search."""
        if self.matrix is None or self.centroids is None:
            raise RuntimeError("IVFIndex.build() must run before search()")
        if len(self.chunks) == 0 or k <= 0:
            return []
        probe = self.nprobe if nprobe is None else nprobe
        q = np.asarray(query_vec, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(q))
        q = q / norm if norm > 0 else q
        c_sims = self.centroids @ q
        probe = max(1, min(probe, self.centroids.shape[0]))
        clusters = np.argsort(-c_sims, kind="stable")[:probe]
        rows = [r for c in clusters for r in self.inverted[int(c)]]
        if not rows:
            return []
        # ACL before ranking, same contract as VectorStore
        if allowed_roles is not None:
            rows = [r for r in rows
                    if allowed_roles.intersection(self.chunks[r].allowed_roles)]
        if not rows:
            return []
        scores = self.matrix[rows].astype(np.float64) @ q
        order = np.argsort(-scores, kind="stable")
        out: list[tuple[Chunk, float]] = []
        for r in order[:k]:
            out.append((self.chunks[rows[int(r)]], float(scores[int(r)])))
        return out

    def recall_at_k(self, query_vecs: np.ndarray, k: int,
                    brute: VectorStore,
                    allowed_roles: set[str] | None = None) -> float:
        """Mean fraction of the brute-force top-k that IVF also returns."""
        if len(query_vecs) == 0:
            return 0.0
        hits = 0
        for q in query_vecs:
            truth = {c.chunk_id for c, _ in brute.search(q, k, allowed_roles)}
            if not truth:
                continue
            found = {c.chunk_id for c, _ in self.search(q, k, allowed_roles)}
            hits += len(truth & found) / len(truth)
        return hits / len(query_vecs)
