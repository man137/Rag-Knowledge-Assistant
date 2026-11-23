# PROGRESS.md — build checklist

Working rule: build in milestone order, run that milestone's pytest immediately,
fix every failure before ticking, never leave a failing test. If context runs
low, re-read this file and resume at the first unticked item.

## M0 — scaffolding

- [x] repo directory tree created
- [x] PROGRESS.md (this file)
- [x] pytest.ini
- [x] requirements.txt (pins resolved OK: pip install --dry-run exit 0)
- [x] requirements-dev.txt
- [x] conftest.py (root)

## M1 — config, loaders, cleaning, chunking, data/docs

- [x] rag/__init__.py
- [x] rag/config.py
- [x] rag/loaders.py
- [x] rag/cleaning.py
- [x] rag/chunking.py
- [x] data/docs/travel_expense_policy.md (roles: all; 904 words)
- [x] data/docs/event_registration_faq.md (all; 837 words; refund rules + marked
      synthetic poisoned appendix used only by injection probes)
- [x] data/docs/venue_booking_sop.md (staff, manager; 861 words)
- [x] data/docs/visa_travel_advisory_faq.md (all; 750 words)
- [x] data/docs/support_escalation_sop.md (staff, manager; 750 words)
- [x] data/docs/hr_compensation_leave_policy.md (hr; 773 words)
- [x] word counts of all six docs verified 700-1200 (body only)
- [x] tests/test_config.py, tests/test_loaders.py, tests/test_cleaning.py,
      tests/test_chunking.py — pytest green (35 passed)

## M2 — embedder, vector_store (+IVF), bm25, hybrid, reranker, mmr

- [x] rag/embedder.py (Protocol, SentenceTransformerEmbedder, FakeEmbedder)
- [x] rag/vector_store.py (brute force + ACL-before-ranking + save/load + IVF)
- [x] rag/bm25.py (tokeniser, inverted index, BM25, save/load)
- [x] rag/hybrid.py (Retriever interface, RetrievalResult, RRF, min-max fusion)
- [x] rag/reranker.py (cross-encoder behind flag + FakeReranker)
- [x] rag/mmr.py
- [ ] tests/fakes.py (comes with M3: FakeLLM lives in rag.generator)
- [x] tests/test_embedder.py, test_vector_store.py, test_ivf.py, test_bm25.py,
      test_hybrid.py, test_reranker.py, test_mmr.py — pytest green
      (80 passed, 1 deselected)

## M3 — security, prompt, prompts/, generator, pipeline

- [x] rag/security.py (roles, PII redaction, injection heuristics, length cap)
- [x] rag/prompt.py (template loading, token budget, chunk blocks)
- [x] prompts/prompt_v1.txt, prompt_v2.txt, prompt_v3.txt
- [x] rag/generator.py (LLMProvider, Ollama, OpenAI-compat, FakeLLM, retries,
      citation parsing; Azure lazy)
- [x] rag/pipeline.py (RAGPipeline.ask / ask_stream, timings, mode-aware
      no-answer thresholds, fake_tuned calibration)
- [x] tests/fakes.py (FakeEmbedder/FakeLLM/FakeReranker + context-citing llm)
- [x] tests/test_security.py, test_prompt.py, test_generator.py,
      test_pipeline.py — pytest green (127 passed, 1 deselected)

## M4 — evaluation + eval set + scripts

- [x] rag/evaluation/__init__.py
- [x] rag/evaluation/metrics.py (recall, MRR, nDCG, groundedness, refusal,
      injection canary, leak rate, citation validity, percentiles)
- [x] rag/evaluation/run_eval.py (chunk sizes 256/512/768 × 4 modes, CSV+MD)
- [x] rag/evaluation/prompt_ab.py (v1/v2/v3 comparison table)
- [x] data/eval/eval_set.jsonl (30/8/6/6; gold keywords verified against gold
      sources by the validator; 2 docs-side injection probes on the marked
      synthetic appendix; canary SECRET-CANARY-8F42)
- [x] scripts/validate_eval_set.py (CLI exit 0: "OK: schema, counts, gold
      keywords and access-control restrictions all verified.")
- [x] tests/test_metrics.py, test_run_eval.py (incl. prompt_ab + validator)
      — pytest green (145 passed, 1 deselected)

## M5 — observability, sql, api, ui, agent

- [x] observability/__init__.py, logging_setup.py (JSON logs, request-id
      contextvar, PII redaction), metrics_store.py (SQLite query_log,
      hashed questions only)
- [x] sql/schema.sql, sql/analytics.sql (6 commented queries; nearest-rank
      percentiles via window functions, noted)
- [x] agent/__init__.py, tools.py (registry, schemas, allowlist, confirm),
      crm_mock.py (SQLite), teams_mock.py (memory + log file), loop.py
      (plan-act-observe, one repair attempt, step limit, trace)
- [x] api/__init__.py, schemas.py, auth.py, main.py (/ask SSE, /ask_sync,
      /ingest, /feedback, /metrics, /health, request-id middleware, global
      JSON error handler, 413 on oversized questions)
- [x] ui/index.html (single file, no CDN, SSE via fetch + ReadableStream,
      feedback, stop button, responsive)
- [x] scripts/ingest.py, ask.py, benchmark.py, export_openapi.py,
      run_agent_demo.py (refund use case, scripted fake by default)
- [x] tests/test_observability.py, test_sql.py, test_tools.py, test_loop.py,
      test_agent_mocks.py, test_api.py, test_smoke.py — pytest green
      (199 passed, 1 deselected)

## M6 — azure_adapters, Docker, CI, Makefile, env

- [x] azure_adapters/__init__.py, azure_openai_llm.py, azure_openai_embedder.py,
      azure_ai_search_retriever.py, graph_sharepoint_loader.py,
      teams_webhook.py, README_AZURE.md (all "NOT TESTED AGAINST LIVE SERVICE")
- [x] Dockerfile (multi-stage, non-root, healthcheck), .dockerignore
- [x] .github/workflows/ci.yml (py3.11, pytest -q, validate_eval_set, ruff)
- [x] Makefile (install/test/ingest/ask/serve/eval/prompt-ab/benchmark/
      agent-demo/openapi/docker-build/lint/clean), .env.example, .gitignore,
      .ruff.toml
- [x] tests/test_azure_adapters.py (19 mocked tests) — pytest green
      (218 passed, 1 deselected) + ruff clean

## M7 — documentation

- [x] README.md (mermaid diagram, 5-command setup, run book, 12-row
      troubleshooting, hand-built vs pre-trained, mocked vs live,
      limitations, 2-minute demo script)
- [x] INTERVIEW_NOTES.md (walkthrough, 12 design decisions, 6 failure modes,
      40 Q&A, 10 interviewer questions, claims-not-to-make)
- [x] JD_MAPPING.md (every JD duty/skill mapped to evidence + honest gap)
- [x] COPILOT_STUDIO_GUIDE.md (6 steps, SSE limitation + /ask_sync,
      SharePoint alternative, quality comparison; UI steps marked
      "verify in current docs")
- [x] RESUME_KIT.md (6 [MEASURED] bullets, pitches, recruiter email with
      blanks, numbers-checklist with exact commands)

## M8 — final verification and zip

- [x] 1. fresh venv install of pinned requirements (or honest report)
      — full pin set installs cleanly with the documented CPU-torch path
      (the default CUDA torch chain exceeds this sandbox's disk);
      `pip check` clean; NOTE: the full default install resolved OK via
      `pip install --dry-run` (exit 0) but ran out of disk mid-install of
      the ~5 GB CUDA wheels, so verification used the requirements.txt
      CPU-torch path instead
- [x] 2. pytest -q — "218 passed, 1 deselected" in BOTH the sandbox
      environment and the fresh venv with the pinned requirements
- [x] 3. python scripts/validate_eval_set.py — "OK: schema, counts, gold
      keywords and access-control restrictions all verified."
- [x] 4. ingest --fake then ask --fake refund question — runs end to end
      (12 chunks indexed; FAQ sources retrieved; fake answer + per-rank
      provenance printed)
- [x] 5. API smoke: /health (index_size=12, auth_enabled), /ask SSE event
      order sources -> token x2 -> done, /feedback up recorded (unknown id
      -> 404), /metrics (total_queries=1, up=1), missing key -> 401
- [x] 6. python scripts/run_agent_demo.py (fake LLM) — scenario A answers
      with a real cited chunk id; scenario B: search -> no_answer ->
      CRM ticket (confirm=true) -> #support-escalations post -> final
      answer; full timestamped trace
- [x] 7. run_eval --fake --limit 10 — 12 rows (3 sizes x 4 modes) written
      to eval_out/results.csv + results.md (HARNESS CHECK ONLY, not real
      results); prompt_ab.md also written
- [x] 8. grep TODO / "..." / NotImplemented — zero placeholder hits (the
      only matches are typing-Protocol `...` stubs and the PROGRESS line
      itself; no TODO, no NotImplemented)
- [x] 9. numeric claims audit — README/RESUME_KIT test counts corrected to
      the verified "218 passed, 1 deselected"; every resume metric kept as
      [MEASURED: ___] placeholders
- [x] 10. zip (no .venv/caches/indexes/logs/eval_out) to
       /home/z/my-project/download/rag-enterprise-agent.zip (118 files,
       openapi.json included)
- [x] PROGRESS.md fully ticked
