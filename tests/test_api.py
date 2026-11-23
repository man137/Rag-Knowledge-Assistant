"""Tests for the FastAPI layer with the fake pipeline (SSE order, auth, ops)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from observability.metrics_store import MetricsStore
from rag.embedder import FakeEmbedder
from rag.pipeline import chunk_corpus, pipeline_from_chunks
from tests.fakes import context_citing_llm

REFUND = "What is the refund rule for event cancellation?"
HEADERS = {"X-API-Key": "test-key", "X-User-Role": "guest"}


@pytest.fixture()
def app(settings, docs):
    chunks = chunk_corpus(settings, docs)
    metrics = MetricsStore(settings.metrics_db_path)
    # the PIPELINE holds the metrics recorder, so wire the same store into
    # both the pipeline and the app (the API reads it for /metrics)
    pipeline = pipeline_from_chunks(settings, chunks, FakeEmbedder(),
                                    context_citing_llm(),
                                    metrics_store=metrics)
    application = create_app(settings, pipeline=pipeline,
                             metrics_store=metrics)
    yield application
    metrics.close()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for frame in text.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        event, data = None, None
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        events.append((event, data))
    return events


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["index_ready"] is True
    assert body["index_size"] > 0
    assert body["provider"] == "fake"
    assert body["default_mode"] == "hybrid"
    assert body["auth_enabled"] is True


def test_ask_sse_event_order(client):
    with client.stream("POST", "/ask", headers=HEADERS,
                       json={"question": REFUND, "mode": "hybrid", "k": 4}) \
            as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())
    events = _parse_sse(text)
    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert names[-1] == "done"
    assert "token" in names
    assert names.index("sources") < names.index("token") < len(names) - 1
    done = dict(events)["done"]
    assert done["no_answer"] is False
    assert done["citations"]["valid"]
    assert done["citations"]["invalid"] == []
    assert done["user_role"] == "guest"
    assert done["request_id"]
    # the sources never carry the full chunk text over the wire
    sources = dict(events)["sources"]["sources"]
    assert all("text" not in s for s in sources)
    assert all("snippet" in s for s in sources)
    # token events concatenate to the done answer
    tokens = "".join(data["text"] for name, data in events if name == "token")
    assert tokens == done["answer"]


def test_ask_sync_returns_full_json(client):
    response = client.post("/ask_sync", headers=HEADERS,
                           json={"question": REFUND, "mode": "hybrid", "k": 4})
    assert response.status_code == 200
    body = response.json()
    assert body["no_answer"] is False
    assert body["answer"]
    assert body["sources"][0]["source"].endswith("event_registration_faq.md")
    assert all("text" not in s for s in body["sources"])
    assert "timings" in body and "request_id" in body


def test_ask_auth_required(client):
    no_key = client.post("/ask", json={"question": REFUND})
    assert no_key.status_code == 401
    bad_key = client.post("/ask",
                          headers={"X-API-Key": "wrong", "X-User-Role": "guest"},
                          json={"question": REFUND})
    assert bad_key.status_code == 401


def test_auth_disabled_when_api_key_empty(settings, docs):
    chunks = chunk_corpus(settings, docs)
    pipeline = pipeline_from_chunks(settings, chunks, FakeEmbedder(),
                                    context_citing_llm())
    open_settings = settings.replace(api_key="")
    metrics = MetricsStore(settings.metrics_db_path)
    app = create_app(open_settings, pipeline=pipeline, metrics_store=metrics)
    try:
        with TestClient(app) as client:
            assert client.get("/health").json()["auth_enabled"] is False
            response = client.post("/ask_sync", json={"question": REFUND})
            assert response.status_code == 200
    finally:
        metrics.close()


def test_oversized_question_is_413(client, settings):
    long_question = "x" * (settings.max_question_length + 1)
    response = client.post("/ask", headers=HEADERS,
                           json={"question": long_question})
    assert response.status_code == 413
    assert "too long" in response.json()["detail"]


def test_bad_mode_and_validation_are_422(client):
    response = client.post("/ask", headers=HEADERS,
                           json={"question": REFUND, "mode": "magic"})
    assert response.status_code == 422
    response = client.post("/ask", headers=HEADERS,
                           json={"question": "", "mode": "hybrid"})
    assert response.status_code == 422
    assert response.json()["error"] == "request validation failed"


def test_feedback_roundtrip_and_404(client):
    with client.stream("POST", "/ask", headers=HEADERS,
                       json={"question": REFUND}) as response:
        done = dict(_parse_sse("".join(response.iter_text())))["done"]
    rid = done["request_id"]
    ok = client.post("/feedback", headers=HEADERS,
                     json={"request_id": rid, "value": "up"})
    assert ok.status_code == 200 and ok.json()["recorded"] is True
    missing = client.post("/feedback", headers=HEADERS,
                          json={"request_id": "nope", "value": "down"})
    assert missing.status_code == 404
    bad_value = client.post("/feedback", headers=HEADERS,
                            json={"request_id": rid, "value": "sideways"})
    assert bad_value.status_code == 422


def test_metrics_endpoint_summarises_queries(client):
    client.post("/ask_sync", headers=HEADERS, json={"question": REFUND})
    response = client.get("/metrics", headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["total_queries"] >= 1
    assert body["latency_p50_ms"] is not None
    assert "by_mode" in body


def test_metrics_requires_auth(client):
    assert client.get("/metrics").status_code == 401


def test_ingest_upload_rebuilds_index(client, settings):
    upload = ("--\ntitle: Extra Policy\nallowed_roles: [all]\n--\n\n"
              "Refunds for the winter gala follow the standard 50 percent "
              "refund window.")
    response = client.post("/ingest", headers=HEADERS,
                           files={"files": ("extra_policy.md",
                                            upload.encode("utf-8"),
                                            "text/markdown")},
                           data={"rebuild": "true"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "rebuild"
    assert body["documents"] == 7          # 6 corpus docs + 1 upload
    assert body["chunks"] > 0
    assert Path(body["index_dir"]).is_dir()
    # next request reloads the rebuilt index and can answer
    answer = client.post("/ask_sync", headers=HEADERS,
                         json={"question": REFUND})
    assert answer.status_code == 200
    assert answer.json()["sources"]


def test_request_id_middleware_sets_header(client):
    response = client.get("/health", headers={"X-Request-ID": "fixed-id-1"})
    assert response.headers["X-Request-ID"] == "fixed-id-1"
    generated = client.get("/health").headers["X-Request-ID"]
    assert generated and generated != "fixed-id-1"


def test_ui_page_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "RAG Knowledge Assistant" in response.text
    assert "fetch(" in response.text and "AbortController" in response.text


def test_role_header_controls_access(client):
    restricted = {"X-API-Key": "test-key", "X-User-Role": "guest"}
    denied = client.post("/ask_sync", headers=restricted,
                         json={"question":
                               "What is the maternity leave entitlement?"})
    assert denied.status_code == 200
    assert denied.json()["no_answer"] is True     # refused, not leaked
    hr_headers = {"X-API-Key": "test-key", "X-User-Role": "hr"}
    allowed = client.post("/ask_sync", headers=hr_headers,
                          json={"question":
                                "What is the maternity leave entitlement?"})
    assert allowed.json()["no_answer"] is False
