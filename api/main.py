"""FastAPI service: SSE /ask, /ask_sync, /ingest, /feedback, /metrics, /health.

WHAT: exposes the RAG pipeline over HTTP. /ask streams Server-Sent Events in
the strict order sources -> token* -> done|error; /ask_sync returns the whole
response as JSON (for Copilot Studio / Power Automate, which cannot consume
SSE); /ingest rebuilds or extends the index from uploaded files;
/feedback attaches thumbs up/down to a request_id; /metrics summarises the
SQLite query_log; /health reports index size, embedding model, provider and
default mode. CORS comes from env, requests get an X-Request-ID, errors come
back as JSON without stack traces, and oversized questions get 413.
WHY: this service is the integration surface everything else consumes: the
bundled UI, the Copilot Studio custom connector (via openapi.json) and any
Power Automate flow.
MATHS: none.
FAILURE MODE: if the index is missing (ingest never ran), /health reports
index_ready=false and /ask answers 503 with a clear message instead of
crashing. SSE streams that fail mid-answer emit an "error" event so the
client can show something instead of hanging.
INTERVIEW ONE-LINER: "The API is one FastAPI app with SSE streaming, a
non-streaming twin for flows, API-key auth designed to become Entra ID, and
JSON errors — plus /metrics fed by the hashed query log."
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError

from api.auth import role_from_header, verify_api_key
from api.schemas import (
    AskRequest,
    FeedbackRequest,
    IngestResult,
    sse_event,
    strip_source_text,
)
from observability.logging_setup import configure_logging, set_request_id
from observability.metrics_store import MetricsStore
from rag.bm25 import BM25Index
from rag.config import Settings
from rag.embedder import FakeEmbedder, build_embedder
from rag.generator import get_provider
from rag.loaders import load_directory, load_file
from rag.pipeline import PIPELINE_MODES as PIPELINE_MODE_NAMES
from rag.pipeline import (
    build_index_in_memory,
    chunk_corpus,
    fake_tuned,
    pipeline_from_chunks,
    pipeline_from_index,
)
from rag.security import InputTooLongError
from rag.vector_store import VectorStore

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODE = "hybrid"


def load_or_build_pipeline(settings, metrics_store: MetricsStore | None = None):
    """Pipeline from the on-disk index; in fake mode a missing index is
    auto-built in memory from data/docs so the demo can run anywhere."""
    fake_llm = settings.llm_provider == "fake"
    fake_embed = settings.embedding_model == "fake"
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        fake_embed = True
    if fake_embed:
        settings = fake_tuned(settings)
    embedder = FakeEmbedder(seed=settings.seed) if fake_embed \
        else build_embedder(settings)
    index_dir = settings.index_dir_for(embedder.name)
    try:
        return pipeline_from_index(
            settings, embedder, index_dir,
            generator=get_provider(settings, fake=fake_llm),
            metrics_store=metrics_store)
    except FileNotFoundError:
        if not fake_embed:
            raise
        docs = load_directory(settings.data_dir)
        chunks = chunk_corpus(settings, docs)
        log.warning("index missing in %s; building an in-memory fake index "
                    "from %d documents", index_dir, len(docs))
        return pipeline_from_chunks(
            settings, chunks, embedder,
            generator=get_provider(settings, fake=fake_llm),
            metrics_store=metrics_store)


def create_app(settings: Settings | None = None, pipeline=None,
               metrics_store: MetricsStore | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    if not settings.api_key:
        log.warning("API_KEY is empty: authentication is DISABLED. "
                    "Set API_KEY before exposing this service.")

    app = FastAPI(
        title="rag-enterprise-agent",
        version="1.0.0",
        description=("Hand-built RAG pipeline over a synthetic events & "
                     "travel knowledge base: hybrid retrieval (dense + BM25 "
                     "+ RRF), role-based access control, citation validation "
                     "and agent tool-calls."),
    )
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.metrics_store = metrics_store or MetricsStore(settings.metrics_db_path)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "X-User-Role", "X-Request-ID",
                           "Content-Type"],
        )

    # ------------------------------------------------------------------ #
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        set_request_id(rid)   # propagates into every JSON log line
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    def require_api_key(x_api_key: str = Header(default="")) -> None:
        if not verify_api_key(x_api_key, settings.api_key):
            raise HTTPException(status_code=401,
                                detail="invalid or missing X-API-Key header")

    def get_pipeline():
        if app.state.pipeline is None:
            try:
                app.state.pipeline = load_or_build_pipeline(
                    settings, app.state.metrics_store)
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"index not built yet: {exc}. "
                           "Run scripts/ingest.py or POST /ingest first.") from exc
        return app.state.pipeline

    def _check_question(question: str) -> None:
        if len(question) > settings.max_question_length:
            raise HTTPException(
                status_code=413,
                detail=f"question too long: {len(question)} characters, "
                       f"limit is {settings.max_question_length}")

    def _check_mode(mode: str) -> None:
        if mode not in PIPELINE_MODE_NAMES:
            raise HTTPException(
                status_code=422,
                detail=f"mode must be one of {PIPELINE_MODE_NAMES}")

    # ------------------------------------------------------------------ #
    @app.get("/health")
    def health() -> dict[str, Any]:
        index_size = 0
        index_ready = False
        try:
            index_size = len(get_pipeline().dense.store)
            index_ready = True
        except HTTPException:
            pass
        if settings.llm_provider == "fake":
            embedding_model = FakeEmbedder().name
        else:
            embedding_model = settings.embedding_model
        return {
            "status": "ok",
            "index_ready": index_ready,
            "index_size": index_size,
            "embedding_model": embedding_model,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "default_mode": DEFAULT_MODE,
            "prompt_version": settings.prompt_version,
            "auth_enabled": bool(settings.api_key),
        }

    @app.post("/ask", dependencies=[Depends(require_api_key)])
    def ask(body: AskRequest,
            x_user_role: str = Header(default="")) -> StreamingResponse:
        _check_question(body.question)
        _check_mode(body.mode)
        pipe = get_pipeline()
        role = role_from_header(x_user_role)
        rid = uuid.uuid4().hex[:12]

        def event_stream():
            try:
                for event in pipe.ask_stream(
                        body.question, mode=body.mode, k=body.k,
                        user_role=role, prompt_version=body.prompt_version,
                        request_id=rid):
                    if event["type"] == "sources":
                        data = {"request_id": rid,
                                "sources": strip_source_text(event["sources"])}
                        yield sse_event("sources", data)
                    elif event["type"] == "token":
                        yield sse_event("token", {"text": event["text"]})
                    elif event["type"] == "done":
                        response = event["response"]
                        response["sources"] = strip_source_text(
                            response["sources"])
                        response["user_role"] = role
                        yield sse_event("done", response)
            except Exception as exc:  # noqa: BLE001 - SSE needs an event
                log.exception("stream failed for request %s", rid)
                yield sse_event("error", {"request_id": rid,
                                         "message": f"{type(exc).__name__}: "
                                                    f"{exc}"})

        return StreamingResponse(
            event_stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "X-Request-ID": rid})

    @app.post("/ask_sync", dependencies=[Depends(require_api_key)])
    def ask_sync(body: AskRequest,
                 x_user_role: str = Header(default="")) -> dict[str, Any]:
        """Non-streaming twin of /ask for connectors and flows (SSE is hard
        to consume from Power Automate / Copilot Studio)."""
        _check_question(body.question)
        _check_mode(body.mode)
        pipe = get_pipeline()
        role = role_from_header(x_user_role)
        rid = uuid.uuid4().hex[:12]
        try:
            response = pipe.ask(body.question, mode=body.mode, k=body.k,
                                user_role=role,
                                prompt_version=body.prompt_version,
                                request_id=rid)
        except InputTooLongError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        payload = response.to_dict()
        payload["sources"] = strip_source_text(payload["sources"])
        payload["user_role"] = role
        return payload

    # ------------------------------------------------------------------ #
    @app.post("/ingest", dependencies=[Depends(require_api_key)])
    async def ingest(files: list[UploadFile] = File(...),
                     rebuild: bool = Form(False)) -> IngestResult:
        """Upload .md/.txt/.pdf files and (re)build the index.
        rebuild=true: fresh index from data/docs + the uploads.
        rebuild=false (incremental): load the persisted index for this
        embedder, add the uploaded chunks to it, and re-save; when no
        persisted index exists yet, incremental degrades to a full rebuild
        and says so in the response mode."""
        with tempfile.TemporaryDirectory(prefix="rag-ingest-") as tmpdir:
            upload_docs = []
            for upload in files:
                path = Path(tmpdir) / (upload.filename or "upload.txt")
                path.write_bytes(await upload.read())
                upload_docs.extend(load_file(path))
            if not upload_docs:
                raise HTTPException(status_code=422,
                                    detail="no readable documents in upload")
            fake_embed = settings.embedding_model == "fake" or settings.llm_provider == "fake"
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                fake_embed = True
            embedder = (FakeEmbedder(seed=settings.seed)
                        if fake_embed
                        else build_embedder(settings))
            eff_settings = fake_tuned(settings) if fake_embed else settings
            index_dir = eff_settings.index_dir_for(embedder.name)

            persisted = (index_dir / "vectors.npy").is_file()
            if not rebuild and persisted:
                # incremental: extend the persisted index with upload chunks
                store = VectorStore.load(index_dir)
                corpus_docs = load_directory(eff_settings.data_dir)
                known_sources = {c.source for c in store.chunks}
                fresh_corpus = [d for d in corpus_docs
                                if d.source not in known_sources]
                new_docs = fresh_corpus + upload_docs
                new_chunks = chunk_corpus(eff_settings, new_docs)
                vectors = embedder.encode_documents(
                    [c.text for c in new_chunks]) if new_chunks \
                    else np.zeros((0, embedder.dim), dtype=np.float32)
                store.add(new_chunks, vectors)
                bm25 = BM25Index()
                bm25.build(store.chunks)
                mode = "incremental"
                docs_count = len(new_docs)
                chunks_count = len(store.chunks)
            else:
                # rebuild (or first ingest): corpus + uploads from scratch
                mode = "rebuild"
                docs = load_directory(eff_settings.data_dir) + upload_docs
                chunks = chunk_corpus(eff_settings, docs)
                store, bm25 = build_index_in_memory(eff_settings, chunks,
                                                    embedder)
                docs_count = len(docs)
                chunks_count = len(chunks)
            store.save(index_dir)
            bm25.save(index_dir / "bm25.json")
        app.state.pipeline = None   # force reload on next request
        log.info("ingest complete: mode=%s total_chunks=%d", mode, chunks_count)
        return IngestResult(documents=docs_count, chunks=chunks_count,
                            mode=mode, index_dir=str(index_dir))

    @app.post("/feedback", dependencies=[Depends(require_api_key)])
    def feedback(body: FeedbackRequest) -> dict[str, Any]:
        recorded = app.state.metrics_store.record_feedback(
            body.request_id, body.value)
        if not recorded:
            raise HTTPException(
                status_code=404,
                detail=f"no query logged with request_id {body.request_id!r}")
        return {"recorded": True, "request_id": body.request_id,
                "value": body.value}

    @app.get("/metrics", dependencies=[Depends(require_api_key)])
    def metrics_summary() -> dict[str, Any]:
        return app.state.metrics_store.summary()

    # ------------------------------------------------------------------ #
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request,
                                 exc: RequestValidationError):
        return JSONResponse(status_code=422, content={
            "error": "request validation failed",
            "details": exc.errors(),
        })

    @app.exception_handler(ValidationError)
    async def pydantic_handler(request: Request, exc: ValidationError):
        return JSONResponse(status_code=422, content={
            "error": "request validation failed",
            "details": exc.errors(),
        })

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        # never leak stack traces to clients; the full trace is in the JSON log
        log.exception("unhandled error on %s %s", request.method,
                      request.url.path)
        return JSONResponse(status_code=500,
                            content={"error": "internal server error"})

    @app.get("/")
    def index_page():
        return FileResponse(REPO_ROOT / "ui" / "index.html")

    return app


# `uvicorn api.main:app` binds the production path
app = create_app()
