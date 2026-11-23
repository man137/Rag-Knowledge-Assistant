"""Configuration for the whole project, loaded from environment variables.

WHAT: one frozen dataclass `Settings` holding every path, seed, model name,
chunking parameter, retrieval parameter, threshold and credential the project
uses, plus `Settings.from_env()` to populate it from the environment (and an
optional .env file parsed by hand — no python-dotenv dependency).
WHY: a single immutable configuration object means every component (ingest,
retrieval, API, agent, evaluation) shares the same tuned values, and tests can
build isolated Settings with `dataclasses.replace`.
MATHS: none.
FAILURE MODE: an unset env var falls back to the default silently, so a typo
in a variable name (e.g. CHUNK_OVERLAP vs CHUNK_OVERLAPS) is ignored. Validate
in `__post_init__` catches impossible values, not wrong names — always print
the effective settings at startup (scripts do).
INTERVIEW ONE-LINER: "One frozen dataclass loaded from env, validated in
__post_init__, so the whole system has a single source of truth for tuning."
"""

from __future__ import annotations

import dataclasses
import random
import re
from pathlib import Path
from typing import Any

# Env var names tried (in order) for fields whose name does not map directly.
_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "llm_model": ("LLM_MODEL", "OLLAMA_MODEL", "OPENAI_MODEL"),
}


def _load_dotenv(path: str | None) -> dict[str, str]:
    """Hand-rolled .env parser. Real environment variables take precedence
    over values from the file. Only KEY=VALUE lines are understood."""
    if not path:
        return {}
    values: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            # setdefault: first wins, matching common .env semantics
            values.setdefault(key, value)
    return values


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclasses.dataclass(frozen=True)
class Settings:
    """All tunables and paths. Frozen: components receive it and cannot
    mutate shared state accidentally."""

    # ---- paths (relative to the repo root by default) ----
    data_dir: str = "data/docs"
    index_dir: str = "indexes"
    cache_dir: str = "cache/embeddings"
    logs_dir: str = "logs"
    prompts_dir: str = "prompts"
    eval_set_path: str = "data/eval/eval_set.jsonl"
    eval_out_dir: str = "eval_out"
    metrics_db_path: str = "logs/metrics.db"
    crm_db_path: str = "logs/crm_mock.db"
    teams_log_path: str = "logs/teams_messages.log"

    # ---- reproducibility ----
    seed: int = 42

    # ---- embedding ----
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache: bool = True

    # ---- chunking ----
    chunk_size: int = 500          # tokens per chunk (approximate word count)
    chunk_overlap: int = 80        # tokens shared between consecutive chunks
    chunking_strategy: str = "sentence"   # "fixed" | "sentence"

    # ---- retrieval ----
    top_k: int = 4                 # chunks finally shown to the user
    candidate_k: int = 20          # candidates fetched per retriever before fusion
    rrf_k: int = 60                # Reciprocal Rank Fusion constant
    rerank_enabled: bool = False   # cross-encoder reranking behind a flag
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 20         # candidates sent to the reranker
    mmr_lambda: float = 0.7        # relevance vs diversity trade-off

    # ---- no-answer thresholds (mode-aware, see rag/pipeline.py) ----
    # Dense: cosine/dot of the top chunk (max 1.0).
    no_answer_threshold: float = 0.35
    # BM25: absolute BM25 score of the top chunk (scale depends on corpus).
    no_answer_threshold_bm25: float = 2.0
    # Reranker: cross-encoder score of the top chunk.
    no_answer_threshold_rerank: float = 0.3

    # ---- generation ----
    llm_provider: str = "ollama"   # fake | ollama | openai | azure
    llm_model: str = "qwen2.5:3b"
    ollama_url: str = "http://localhost:11434"
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-06-01"  # VERSION-SENSITIVE: Azure
    # OpenAI REST API versions change quarterly; if requests are rejected,
    # set AZURE_OPENAI_API_VERSION to a version listed for your resource.
    azure_openai_deployment: str = ""
    llm_timeout_s: float = 60.0
    llm_retries: int = 2
    llm_backoff_s: float = 0.5
    temperature: float = 0.2
    max_output_tokens: int = 512
    context_token_budget: int = 3000   # max prompt tokens incl. question+context

    # ---- security ----
    api_key: str = ""              # empty = auth disabled (dev only; warned)
    cors_origins: tuple[str, ...] = ("http://localhost:8000",)
    max_question_length: int = 1000
    strip_injection: bool = True    # strip injection phrases found in chunks
    redact_pii_in_context: bool = True
    groundedness_threshold: float = 0.6

    # ---- logging / prompts ----
    log_level: str = "INFO"
    prompt_version: str = "v2"

    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls, dotenv: str | None = ".env") -> "Settings":
        """Build Settings from environment variables. `dotenv` may point to a
        .env file (skipped when None) whose values are used only for variables
        missing from the real environment."""
        file_values = _load_dotenv(dotenv)
        kwargs: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            names = _ENV_ALIASES.get(field.name, (field.name.upper(),))
            raw: str | None = None
            for name in names:
                if name in file_values:
                    raw = file_values[name]
                if name in __import__("os").environ:
                    raw = __import__("os").environ[name]
                if raw is not None:
                    break
            if raw is None:
                continue
            kwargs[field.name] = _coerce(raw, field.type)
        return cls(**kwargs)

    def replace(self, **changes: Any) -> "Settings":
        """dataclasses.replace wrapper (frozen dataclasses are immutable)."""
        return dataclasses.replace(self, **changes)

    def apply_seed(self) -> None:
        """Seed `random` and NumPy from settings so every run is reproducible
        (BM25 tie-breaks, FakeEmbedder jitter, IVF k-means init, backoff)."""
        random.seed(self.seed)
        try:
            import numpy as np

            np.random.seed(self.seed % (2**32))
        except ImportError:  # pragma: no cover - numpy is a pinned dependency
            pass

    def embedder_slug(self, embedder_name: str) -> str:
        """Filesystem-safe directory name for an embedder's index."""
        return re.sub(r"[^A-Za-z0-9._-]+", "_", embedder_name)

    def index_dir_for(self, embedder_name: str) -> Path:
        """Indexes are stored per embedder so fake/real never collide."""
        return Path(self.index_dir) / self.embedder_slug(embedder_name)

    def __post_init__(self) -> None:
        if not 0 < self.chunk_overlap < self.chunk_size:
            raise ValueError(
                f"chunk_overlap must be in (0, chunk_size), got "
                f"overlap={self.chunk_overlap} size={self.chunk_size}"
            )
        if self.top_k <= 0 or self.candidate_k < self.top_k:
            raise ValueError("need 0 < top_k <= candidate_k")
        if self.chunking_strategy not in ("fixed", "sentence"):
            raise ValueError(f"unknown chunking_strategy {self.chunking_strategy!r}")
        if self.llm_provider not in ("fake", "ollama", "openai", "azure"):
            raise ValueError(f"unknown llm_provider {self.llm_provider!r}")
        if self.rrf_k <= 0 or self.seed < 0:
            raise ValueError("rrf_k and seed must be positive")
        for name in ("no_answer_threshold", "no_answer_threshold_bm25",
                     "no_answer_threshold_rerank"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


def _coerce(raw: str, annotation: Any) -> Any:
    """Convert an env string into the annotated type. Annotations are strings
    because of `from __future__ import annotations`."""
    ann = str(annotation)
    if ann == "bool":
        return _parse_bool(raw)
    if ann == "int":
        return int(raw)
    if ann == "float":
        return float(raw)
    if ann.startswith("tuple"):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return tuple(parts)
    return raw  # str
