"""Tests for rag.config."""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import pytest

from rag.config import Settings

ENV_ALIASES = ("LLM_MODEL", "OLLAMA_MODEL", "OPENAI_MODEL")


def _scrub_env(monkeypatch):
    """Remove every env var that can influence Settings."""
    names = [f.name.upper() for f in dataclasses.fields(Settings)]
    names.extend(ENV_ALIASES)
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_defaults_without_env(monkeypatch):
    _scrub_env(monkeypatch)
    s = Settings.from_env(dotenv=None)
    assert s.chunk_size == 500 and s.chunk_overlap == 80
    assert s.chunking_strategy == "sentence"
    assert s.top_k == 4 and s.candidate_k == 20 and s.rrf_k == 60
    assert s.no_answer_threshold == 0.35
    assert s.no_answer_threshold_bm25 == 2.0
    assert s.llm_provider == "ollama" and s.llm_model == "qwen2.5:3b"
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.cors_origins == ("http://localhost:8000",)
    assert s.api_key == ""  # empty = auth disabled in dev (warned at runtime)
    assert s.max_question_length == 1000
    assert s.context_token_budget == 3000


def test_env_overrides(monkeypatch):
    _scrub_env(monkeypatch)
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3:8b")
    monkeypatch.setenv("API_KEY", "k-123")
    monkeypatch.setenv("MAX_QUESTION_LENGTH", "80")
    s = Settings.from_env(dotenv=None)
    assert s.chunk_size == 256
    assert s.rerank_enabled is True
    assert s.cors_origins == ("http://a.com", "http://b.com")
    assert s.llm_model == "llama3:8b"
    assert s.api_key == "k-123"
    assert s.max_question_length == 80


def test_dotenv_used_when_env_missing(tmp_path, monkeypatch):
    _scrub_env(monkeypatch)
    envfile = tmp_path / ".env"
    envfile.write_text("# comment\nSEED=99\nexport TEMPERATURE=0.5\nBADLINE\n")
    s = Settings.from_env(dotenv=str(envfile))
    assert s.seed == 99 and s.temperature == 0.5
    # the real environment wins over the .env file
    monkeypatch.setenv("SEED", "7")
    s2 = Settings.from_env(dotenv=str(envfile))
    assert s2.seed == 7


def test_validation_rejects_impossible_values():
    with pytest.raises(ValueError):
        Settings(chunk_size=10, chunk_overlap=10)   # overlap must be < size
    with pytest.raises(ValueError):
        Settings(top_k=0)
    with pytest.raises(ValueError):
        Settings(chunking_strategy="nonsense")
    with pytest.raises(ValueError):
        Settings(llm_provider="claude")
    with pytest.raises(ValueError):
        Settings(rrf_k=0)
    Settings()  # defaults must validate


def test_frozen_and_replace():
    s = Settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.chunk_size = 1  # type: ignore[misc]
    s2 = s.replace(chunk_size=256)
    assert s2.chunk_size == 256 and s.chunk_size == 500


def test_apply_seed_is_deterministic():
    s = Settings(seed=123)
    s.apply_seed()
    first = [random.random() for _ in range(5)]
    s.apply_seed()
    second = [random.random() for _ in range(5)]
    assert first == second


def test_index_dir_per_embedder():
    s = Settings()
    assert s.embedder_slug("BAAI/bge-small-en-v1.5") == "BAAI_bge-small-en-v1.5"
    assert s.index_dir_for("fake-embedder") == Path(s.index_dir) / "fake-embedder"
