"""The RAG pipeline: one orchestrated, timed, auditable ask() call.

WHAT: RAGPipeline.ask(question, mode, k, user_role, prompt_version) runs:
length check -> injection scan of the question -> role allow-set ->
retrieve -> optional rerank -> no-answer check -> prepare context (strip
injection, redact PII) -> build prompt under the token budget (dropping the
lowest-ranked chunks first) -> generate (stream or not) -> validate citations
-> record a metrics row. ask_stream() exposes the same flow as an event
generator (sources / token* / done) for the SSE API.
WHY: the pipeline is the single place where retrieval, security, prompting
and generation meet; per-stage timings make /metrics and the eval harness
meaningful (p50/p95 latency, time-to-first-token, no-answer rate).
MATHS: mode-aware no-answer thresholds (see rag.config):
- dense      : top cosine score            vs no_answer_threshold (max 1.0)
- bm25       : top absolute BM25 score     vs no_answer_threshold_bm25
- hybrid     : DENSE score of the top fused chunk vs no_answer_threshold —
  RRF scores are rank-only (every list has a rank 1) so they can never express
  "nothing is relevant"
- hybrid_rerank: reranker score of the top chunk vs no_answer_threshold_rerank.
When the top confidence is below threshold the LLM call is SKIPPED and the
canonical refusal sentence from rag.prompt is returned: cheaper, safer, and
measurable as a no_answer rate in analytics.
FAILURE MODE: below-threshold retrieval yields the refusal even when the
question IS in scope — a mis-calibrated threshold turns into a high no-answer
rate that run_eval.py exposes per mode, which is exactly why the threshold is
configurable and evaluated rather than guessed.
INTERVIEW ONE-LINER: "ask() is a timed state machine — retrieve, maybe
rerank, budget-check the prompt, generate, validate citations — that refuses
early when retrieval confidence is below threshold instead of letting the
model hallucinate from weak context."
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

from rag import security
from rag.bm25 import BM25Index
from rag.chunking import Chunk
from rag.cleaning import tokens
from rag.embedder import Embedder
from rag.generator import LLMProvider, get_provider
from rag.hybrid import BM25Retriever, DenseRetriever, HybridRetriever, RetrievalResult
from rag.prompt import REFUSAL, build_messages
from rag.reranker import build_reranker
from rag.vector_store import VectorStore

log = logging.getLogger(__name__)

PIPELINE_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class SourceInfo:
    """What the API returns per retrieved chunk."""

    chunk_id: str
    source: str
    title: str
    score: float
    snippet: str
    text: str                       # full chunk text (eval groundedness; the
                                    # API strips this before responding)
    ranks: dict[str, int]
    scores: dict[str, float]

    @classmethod
    def from_result(cls, res: RetrievalResult, snippet_len: int = 200) -> "SourceInfo":
        snippet = " ".join(res.chunk.text.split())[:snippet_len]
        return cls(chunk_id=res.chunk.chunk_id, source=res.chunk.source,
                   title=res.chunk.title, score=round(res.score, 6),
                   snippet=snippet, text=res.chunk.text,
                   ranks=dict(res.ranks),
                   scores={k: round(v, 6) for k, v in res.scores.items()})

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "source": self.source,
                "title": self.title, "score": self.score, "snippet": self.snippet,
                "text": self.text,
                "ranks": self.ranks, "scores": self.scores}


@dataclass
class RAGResponse:
    """Everything one ask() produced, for the API and the eval harness."""

    answer: str
    sources: list[SourceInfo] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    ttft_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    no_answer: bool = False
    valid_citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    injection_flags: list[str] = field(default_factory=list)
    dropped_chunks: list[str] = field(default_factory=list)
    request_id: str = ""
    mode: str = ""
    prompt_version: str = ""
    user_role: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "ttft_ms": round(self.ttft_ms, 3),
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "no_answer": self.no_answer,
            "citations": {"valid": self.valid_citations,
                          "invalid": self.invalid_citations},
            "injection_flags": self.injection_flags,
            "dropped_chunks": self.dropped_chunks,
            "request_id": self.request_id,
            "mode": self.mode, "prompt_version": self.prompt_version,
            "user_role": self.user_role,
        }


def question_hash(question: str) -> str:
    """Questions are never stored raw in metrics; a sha1 prefix keeps the
    analytics useful (grouping repeated questions) without keeping content."""
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]


class RAGPipeline:
    """Orchestrates retrieval, security, prompting and generation."""

    def __init__(self, settings, embedder: Embedder,
                 dense: DenseRetriever, bm25: BM25Retriever,
                 generator: LLMProvider, reranker=None,
                 metrics_store: Any = None) -> None:
        self.settings = settings
        self.embedder = embedder
        self.dense = dense
        self.bm25 = bm25
        self.generator = generator
        self.reranker = reranker
        self.metrics_store = metrics_store
        self.snippet_len = 200

    # ------------------------------------------------------------ #
    def ask(self, question: str, mode: str | None = None, k: int | None = None,
            user_role: str = "guest", prompt_version: str | None = None,
            request_id: str | None = None) -> RAGResponse:
        """Non-streaming ask: returns the full RAGResponse."""
        response: RAGResponse | None = None
        for event in self._ask_events(question, mode=mode, k=k,
                                       user_role=user_role,
                                       prompt_version=prompt_version,
                                       request_id=request_id, stream=False):
            if event["type"] == "done":
                response = _response_from_dict(event["response"])
        if response is None:  # pragma: no cover - generator always ends here
            raise RuntimeError("pipeline produced no done event")
        return response

    def ask_stream(self, question: str, mode: str | None = None,
                   k: int | None = None, user_role: str = "guest",
                   prompt_version: str | None = None,
                   request_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Streaming ask: yields sources / token / done events (SSE source)."""
        yield from self._ask_events(question, mode=mode, k=k, user_role=user_role,
                                     prompt_version=prompt_version,
                                     request_id=request_id, stream=True)

    # ------------------------------------------------------------ #
    def _ask_events(self, question: str, mode: str | None, k: int | None,
                    user_role: str, prompt_version: str | None,
                    request_id: str | None,
                    stream: bool) -> Iterator[dict[str, Any]]:
        settings = self.settings
        started = time.perf_counter()
        rid = request_id or uuid.uuid4().hex[:12]
        mode = mode or "hybrid"
        if mode not in PIPELINE_MODES:
            raise ValueError(f"unknown mode {mode!r}; use {PIPELINE_MODES}")
        effective_mode = mode
        if mode == "hybrid_rerank" and self.reranker is None:
            # reranking requested but not configured: degrade to hybrid,
            # visibly, instead of silently pretending to rerank
            log.warning("mode=hybrid_rerank but no reranker configured; "
                        "falling back to hybrid")
            effective_mode = "hybrid"
        k = k or settings.top_k
        prompt_version = prompt_version or settings.prompt_version
        user_role = security.normalise_role(user_role)

        security.check_question_length(question, settings.max_question_length)
        q_flags = security.detect_injection(question)
        injection_flags = [f"question:{p}" for p in q_flags]
        roles = security.effective_allowed_roles(user_role)

        timings: dict[str, float] = {}

        # ---- retrieve ----
        t0 = time.perf_counter()
        retriever = HybridRetriever(self.dense, self.bm25,
                                    mode="hybrid" if effective_mode == "hybrid_rerank"
                                    else effective_mode,
                                    rrf_k=settings.rrf_k,
                                    candidate_k=settings.candidate_k)
        if effective_mode == "hybrid_rerank":
            fetch_k = max(settings.rerank_top_n, k)
        elif effective_mode in ("hybrid", "hybrid_minmax"):
            fetch_k = max(settings.candidate_k, k)
        else:
            fetch_k = k
        results = retriever.retrieve(question, fetch_k, roles)
        timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000

        # ---- optional rerank ----
        timings["rerank_ms"] = 0.0
        if effective_mode == "hybrid_rerank":
            t1 = time.perf_counter()
            results = list(self.reranker.rerank(question, results, top_n=k))
            timings["rerank_ms"] = (time.perf_counter() - t1) * 1000

        # ---- no-answer check (mode-aware, see module docstring) ----
        threshold = {
            "dense": settings.no_answer_threshold,
            "bm25": settings.no_answer_threshold_bm25,
            "hybrid": settings.no_answer_threshold,
            "hybrid_rerank": settings.no_answer_threshold_rerank,
        }[effective_mode]

        def top_confidence(res: Sequence[RetrievalResult]) -> float:
            if not res:
                return 0.0
            if effective_mode == "dense":
                return res[0].score
            if effective_mode == "bm25":
                return res[0].score
            if effective_mode == "hybrid_rerank":
                return res[0].scores.get("reranker", 0.0)
            # hybrid: confident if any top retrieved candidate meets dense threshold
            # OR has a strong BM25 lexical match (essential for names, IDs, codes)
            max_dense = max((r.scores.get("dense") or 0.0) for r in res[:k])
            max_bm25 = max((r.scores.get("bm25") or 0.0) for r in res[:k])
            norm_dense = max_dense / (settings.no_answer_threshold or 1.0)
            norm_bm25 = max_bm25 / (settings.no_answer_threshold_bm25 or 1.0)
            return max(norm_dense, norm_bm25) * settings.no_answer_threshold

        no_answer = top_confidence(results) < threshold
        sources = [SourceInfo.from_result(r, self.snippet_len)
                   for r in results[:k]] if not no_answer else []
        yield {"type": "sources", "sources": [s.to_dict() for s in sources]}

        if no_answer:
            answer = REFUSAL
            # skipped stages still report 0.0 so the timings shape is stable
            timings.setdefault("prompt_ms", 0.0)
            timings.setdefault("generate_ms", 0.0)
            done = RAGResponse(
                answer=answer, sources=[], timings=timings, ttft_ms=0.0,
                tokens_in=0, tokens_out=0, no_answer=True,
                injection_flags=injection_flags, dropped_chunks=[],
                request_id=rid, mode=mode, prompt_version=prompt_version,
                user_role=user_role)
            self._record_metrics(done, question, started)
            yield {"type": "done", "response": done.to_dict()}
            return

        # ---- prepare context: strip injection, redact PII ----
        prepared = self._prepare_context(results[:k], injection_flags)

        # ---- prompt under the token budget ----
        t2 = time.perf_counter()
        messages, dropped, tokens_in = build_messages(
            question=question, results=prepared,
            prompt_version=prompt_version, prompts_dir=settings.prompts_dir,
            context_token_budget=settings.context_token_budget,
            max_output_tokens=settings.max_output_tokens)
        timings["prompt_ms"] = (time.perf_counter() - t2) * 1000

        # ---- generate ----
        t3 = time.perf_counter()
        ttft_ms = 0.0
        pieces: list[str] = []
        if stream:
            first = True
            for piece in self.generator.generate(
                    messages, stream=True,
                    max_tokens=settings.max_output_tokens,
                    temperature=settings.temperature):
                if first:
                    ttft_ms = (time.perf_counter() - t3) * 1000
                    first = False
                pieces.append(str(piece))
                yield {"type": "token", "text": str(piece)}
            answer = "".join(pieces)
            if first:  # empty stream: treat as immediate completion
                ttft_ms = (time.perf_counter() - t3) * 1000
        else:
            answer = str(self.generator.generate(
                messages, stream=False, max_tokens=settings.max_output_tokens,
                temperature=settings.temperature))
            ttft_ms = (time.perf_counter() - t3) * 1000
        timings["generate_ms"] = (time.perf_counter() - t3) * 1000

        # ---- citation validation against what was actually sent ----
        context_ids = [r.chunk.chunk_id for r in prepared]
        valid_c, invalid_c = _validate(answer, context_ids)

        done = RAGResponse(
            answer=answer, sources=sources, timings=timings,
            ttft_ms=ttft_ms, tokens_in=tokens_in,
            tokens_out=len(tokens(answer)), no_answer=False,
            valid_citations=valid_c, invalid_citations=invalid_c,
            injection_flags=injection_flags, dropped_chunks=dropped,
            request_id=rid, mode=mode, prompt_version=prompt_version,
            user_role=user_role)
        self._record_metrics(done, question, started)
        yield {"type": "done", "response": done.to_dict()}

    # ------------------------------------------------------------ #
    def _prepare_context(self, results: Sequence[RetrievalResult],
                         injection_flags: list[str]) -> list[RetrievalResult]:
        """Sanitise retrieved chunk texts before they reach the prompt."""
        settings = self.settings
        prepared: list[RetrievalResult] = []
        for res in results:
            text = res.chunk.text
            if settings.strip_injection:
                text, chunk_flags = security.strip_injection(text)
                for flag in chunk_flags:
                    injection_flags.append(
                        f"chunk:{res.chunk.chunk_id}:{flag}")
            if settings.redact_pii_in_context:
                text, _ = security.redact_pii(text)
            if text != res.chunk.text:
                chunk = replace(res.chunk, text=text)
                prepared.append(replace(res, chunk=chunk))
            else:
                prepared.append(res)
        return prepared

    def _record_metrics(self, done: RAGResponse, question: str,
                        started: float) -> None:
        if self.metrics_store is None:
            return
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            self.metrics_store.record_query(
                request_id=done.request_id,
                question_hash=question_hash(question),
                mode=done.mode, prompt_version=done.prompt_version,
                user_role=done.user_role,
                latency_ms=round(latency_ms, 3),
                ttft_ms=round(done.ttft_ms, 3),
                tokens_in=done.tokens_in, tokens_out=done.tokens_out,
                no_answer=1 if done.no_answer else 0,
                invalid_citations=len(done.invalid_citations),
            )
        except Exception:  # noqa: BLE001 - metrics must never break answers
            log.exception("metrics recording failed (ignored)")


def _validate(answer: str, context_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    from rag.generator import validate_citations

    return validate_citations(answer, context_ids)


def _response_from_dict(d: dict[str, Any]) -> RAGResponse:
    sources = [SourceInfo(**s) for s in d["sources"]]
    return RAGResponse(
        answer=d["answer"], sources=sources, timings=d["timings"],
        ttft_ms=d["ttft_ms"], tokens_in=d["tokens_in"],
        tokens_out=d["tokens_out"], no_answer=d["no_answer"],
        valid_citations=d["citations"]["valid"],
        invalid_citations=d["citations"]["invalid"],
        injection_flags=d["injection_flags"],
        dropped_chunks=d["dropped_chunks"], request_id=d["request_id"],
        mode=d["mode"], prompt_version=d["prompt_version"],
        user_role=d["user_role"])


def build_index_in_memory(settings, chunks: Sequence[Chunk],
                         embedder: Embedder) -> tuple[VectorStore, BM25Index]:
    """Embed chunks and build both halves of the hybrid index."""
    vectors = embedder.encode_documents([c.text for c in chunks])
    store = VectorStore(dim=embedder.dim)
    store.add(list(chunks), vectors)
    bm25 = BM25Index()
    bm25.build(list(chunks))
    return store, bm25


def pipeline_from_index(settings, embedder: Embedder, index_dir: str | Path,
                       generator: LLMProvider | None = None,
                       reranker=None,
                       metrics_store: Any = None) -> RAGPipeline:
    """Load a saved index from disk (store + BM25 must share the embedder)."""
    index_dir = Path(index_dir)
    if not (index_dir / "vectors.npy").exists():
        raise FileNotFoundError(
            f"No vector index in {index_dir}; run scripts/ingest.py first.")
    store = VectorStore.load(index_dir)
    bm25 = BM25Index.load(index_dir / "bm25.json")
    generator = generator or get_provider(settings)
    if reranker is None:
        reranker = build_reranker(settings)
    return RAGPipeline(
        settings=settings, embedder=embedder,
        dense=DenseRetriever(embedder, store), bm25=BM25Retriever(bm25),
        generator=generator, reranker=reranker, metrics_store=metrics_store)


def pipeline_from_chunks(settings, chunks: Sequence[Chunk],
                         embedder: Embedder,
                         generator: LLMProvider | None = None,
                         reranker=None,
                         metrics_store: Any = None) -> RAGPipeline:
    """Build an in-memory pipeline straight from chunks (tests, API ingest)."""
    store, bm25 = build_index_in_memory(settings, chunks, embedder)
    generator = generator or get_provider(settings)
    if reranker is None:
        reranker = build_reranker(settings)
    return RAGPipeline(
        settings=settings, embedder=embedder,
        dense=DenseRetriever(embedder, store), bm25=BM25Retriever(bm25),
        generator=generator, reranker=reranker, metrics_store=metrics_store)


def fake_tuned(settings):
    """Calibrate no-answer thresholds for FakeEmbedder-based runs (--fake).
    Thresholds are embedder-specific. 0.04 separates in-scope from unrelated."""
    return settings.replace(no_answer_threshold=0.04)


def chunk_corpus(settings, docs) -> list[Chunk]:
    """Clean + chunk loaded documents with the configured strategy."""
    from rag.chunking import make_chunker
    from rag.cleaning import clean_document

    chunker = make_chunker(settings.chunking_strategy, settings.chunk_size,
                          settings.chunk_overlap)
    chunks: list[Chunk] = []
    for doc in docs:
        cleaned = clean_document(doc)
        chunks.extend(chunker.chunk(cleaned.text, cleaned.source,
                                    cleaned.title, cleaned.page,
                                    cleaned.allowed_roles))
    return chunks
