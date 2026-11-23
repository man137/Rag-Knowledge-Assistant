"""Request/response models and SSE helpers for the API.

WHAT: pydantic models for /ask, /ask_sync, /feedback bodies with strict
bounds (non-empty question, k in 1..20, feedback limited to up/down), plus
the SSE wire-format helper used by main.py.
WHY: validation at the boundary keeps garbage out of the pipeline and gives
clean 422s instead of 500s; the k cap bounds retrieval cost per request.
MATHS: none.
FAILURE MODE: pydantic-only length limits would give a generic 422 for a
30,000-character question; the handler converts over-limit questions to an
explicit 413 with a message, so clients know it is size, not shape.
INTERVIEW ONE-LINER: "Strict request models at the edge: empty questions and
out-of-range k never reach the pipeline, and oversize questions get a 413
with the limit spelled out."
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

PIPELINE_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user question")
    mode: str = Field("hybrid", description=f"One of {PIPELINE_MODES}")
    k: int = Field(4, ge=1, le=20,
                   description="Number of chunks to ground the answer")
    prompt_version: Optional[str] = Field(
        None, description="Prompt version override (v1 | v2 | v3)")


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    value: Literal["up", "down"]


class IngestResult(BaseModel):
    documents: int
    chunks: int
    mode: Literal["rebuild", "incremental"]
    index_dir: str


def sse_event(event: str, data: dict) -> str:
    """One SSE frame: named event plus one JSON data line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def strip_source_text(sources: list[dict]) -> list[dict]:
    """Remove the full chunk text from sources before they leave the service;
    clients get the snippet, the eval harness keeps the text internally."""
    out = []
    for source in sources:
        slim = {k: v for k, v in source.items() if k != "text"}
        out.append(slim)
    return out
