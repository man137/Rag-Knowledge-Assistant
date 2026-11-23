"""Shared pytest fixtures for the whole test-suite.

WHAT: repo-root sys.path bootstrap, an autouse seed fixture, an isolated
`settings` fixture (temp index/log/cache dirs, fake provider, known API key,
env scrubbed) and a `docs` fixture loading the six synthetic corpus files.
WHY: every test must run offline, on a laptop CPU, with no model downloads and
no dependence on the developer's real environment variables.
MATHS: none.
FAILURE MODE: a test that reads the real environment directly (instead of via
`settings`) would silently pick up a developer's .env and flake; fixtures are
the single way to obtain configuration.
INTERVIEW ONE-LINER: "The suite is hermetic: seeded RNGs, temp dirs, fakes for
the model, and env scrubbed, so `pytest -q` is deterministic offline."
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.config import Settings  # noqa: E402

DOCS_DIR = REPO_ROOT / "data" / "docs"

# Environment variables that must never leak from the developer's shell into
# the test-suite; `settings` deletes them all before building Settings.
ISOLATED_ENV_VARS = (
    "LLM_PROVIDER", "LLM_MODEL", "OLLAMA_MODEL", "OPENAI_MODEL",
    "OLLAMA_URL", "OPENAI_BASE_URL", "OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "EMBEDDING_MODEL",
    "SEARCH_ENDPOINT", "SEARCH_INDEX", "SEARCH_KEY",
    "GRAPH_TENANT_ID", "GRAPH_CLIENT_ID", "GRAPH_CLIENT_SECRET",
    "SHAREPOINT_SITE_ID", "SHAREPOINT_FOLDER", "SHAREPOINT_ROLE_MAP",
    "TEAMS_WEBHOOK_URL", "API_KEY", "CORS_ORIGINS", "SEED",
)


@pytest.fixture(autouse=True)
def _seed_everything():
    """Deterministic RNGs for every test, even if code forgets to seed."""
    random.seed(42)
    try:
        import numpy as np

        np.random.seed(42)
    except ImportError:  # pragma: no cover
        pass
    yield


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    """Isolated Settings: real synthetic docs on disk, everything else in a
    temp directory, fake LLM provider, known API key."""
    for var in ISOLATED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("API_KEY", "test-key")
    base = Settings.from_env(dotenv=None)
    return base.replace(
        data_dir=str(DOCS_DIR),
        index_dir=str(tmp_path / "indexes"),
        cache_dir=str(tmp_path / "cache"),
        logs_dir=str(tmp_path / "logs"),
        metrics_db_path=str(tmp_path / "logs" / "metrics.db"),
        crm_db_path=str(tmp_path / "logs" / "crm_mock.db"),
        teams_log_path=str(tmp_path / "logs" / "teams_messages.log"),
        prompts_dir=str(REPO_ROOT / "prompts"),
        eval_set_path=str(REPO_ROOT / "data" / "eval" / "eval_set.jsonl"),
        llm_provider="fake",
        api_key="test-key",
        cors_origins=("http://localhost:8000",),
        # FakeEmbedder cosines were measured on this corpus (see
        # rag/pipeline.py fake_tuned): in-scope questions score 0.09-0.28,
        # unrelated questions stay under ~0.045, so fake-based runs use a
        # dense no-answer threshold of 0.07. BGE-based runs keep the default
        # 0.35 from config.
        no_answer_threshold=0.07,
    )


@pytest.fixture()
def docs():
    """The six synthetic corpus documents, loaded from disk."""
    from rag import loaders

    return loaders.load_directory(DOCS_DIR)
