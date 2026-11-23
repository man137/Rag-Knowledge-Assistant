# RESUME_KIT — copy, personalise, MEASURE, then use

Every metric below appears as `[MEASURED: ...]` on purpose: fill it with
YOUR number from the command next to it, on YOUR machine, before it goes on
a resume. Never ship a bullet with the placeholder, and never invent a
value. Numbers from fake-mode runs are harness checks, not results.

## Project title + one-line ATS description

**rag-enterprise-agent — Enterprise RAG knowledge system and tool-calling agent (Python, NumPy, FastAPI)**

Hand-built hybrid retrieval pipeline (chunking, BM25, dense embeddings, RRF
fusion, optional cross-encoder reranking) with role-based access control,
citation validation, prompt A/B testing, SSE streaming API and an
agent loop with guarded tool-calls.

## Six resume bullets (fill the blanks, tag = JD keywords covered)

1. **Built a complete RAG retrieval core from scratch in Python/NumPy** —
   sentence-aware chunking with verifiable overlap, hand-implemented Okapi
   BM25 (k1/b length normalisation), cosine vector search with
   argpartition top-k and an optional IVF k-means index — with no RAG
   frameworks, so every stage is explainable.
   *[tags: Python, RAG knowledge systems]*
2. **Implemented hybrid retrieval and answer grounding** — fused dense and
   BM25 ranks with Reciprocal Rank Fusion, validated model citations
   `[source:chunk_id]` against the context actually sent, and enforced a
   retrieval-confidence threshold that refuses below-threshold questions
   without calling the LLM — achieving [MEASURED: recall@4 on the eval set =
   ___] on a 50-item evaluation set.
   *[tags: RAG, evaluation of accuracy, prompt design]*
3. **Enforced enterprise access control and data handling** — role-based
   ACL applied BEFORE ranking (−inf score mask before top-k), PII redaction
   in logs and context, prompt-injection phrase detection with a canary
   test proving zero restricted-document leakage for unauthorised roles
   ([MEASURED: access-control leak rate = ___]).
   *[tags: security, data handling, access control]*
4. **Designed and A/B-tested three prompt versions** on a fixed eval set,
   reporting groundedness, refusal accuracy, injection resistance and
   citation validity; wrote a JSON-eval-set validator that runs in CI to
   keep gold answers honest. Prompt v3 (grounding + injection defence +
   format rules) improved [MEASURED: metric = ___] over the minimal v1
   baseline.
   *[tags: prompt design and testing, Git, evaluation]*
5. **Instrumented and served the system** — FastAPI API with SSE streaming
   answers, a non-streaming endpoint for connectors, ingest/feedback/metrics
   endpoints, JSON logs with request-id correlation, and a SQLite query log
   (hashed questions) with SQL analytics for p50/p95 latency by mode:
   [MEASURED: p50 latency = ___ ms, TTFT = ___ ms].
   *[tags: APIs, logging and monitoring, basic SQL, evaluation of latency]*
6. **Hand-wrote a plan-act-observe agent loop** with JSON tool-calls, a
   tool allowlist, schema validation and a confirm=true gate on write
   actions, demonstrating answer-with-citations or CRM-ticket + Teams
   escalation flows; documented Azure equivalents (Azure AI Search hybrid,
   Azure OpenAI, Entra ID auth, Key Vault, Application Insights) as
   swappable adapters tested with mocks.
   *[tags: AI agent workflows, enterprise tool integration, Azure, CRM,
   Teams, SharePoint]*

## Three-line Projects entry

**rag-enterprise-agent** — Python, NumPy, FastAPI, SQLite, Docker; Ollama/Azure OpenAI-compatible.
Hand-built enterprise RAG: hybrid dense+BM25 retrieval with RRF, role-based
ACL before ranking, validated citations, refusal thresholds, SSE API,
SQLite latency metrics and a guarded tool-calling agent loop.
Offline test suite of 218 passing tests (plus 1 integration test deselected
by default) with deterministic fakes; prompt and retrieval
parameters tuned on a validated 50-item evaluation set (numbers measured on
request).

## Skills line (only skills the repo truly shows)

Python · NumPy · RAG (chunking, BM25, dense retrieval, RRF, reranking) ·
Prompt engineering & A/B evaluation · FastAPI / REST / SSE · SQLite / SQL
analytics · Role-based access control & PII redaction · Logging &
monitoring (JSON, request-ids) · Agent tool-calling design · Docker ·
Git/GitHub Actions CI · Azure OpenAI / Azure AI Search / Microsoft Graph /
Teams / Entra ID (adapter-level knowledge, not deployed) · Ollama ·
pytest

## GitHub repo packaging

- **Repo description** (under the title, 350 chars):
  `Enterprise RAG built from scratch: hybrid dense+BM25 retrieval with RRF, role-based ACL, citation validation, refusal thresholds, SSE API, latency metrics, and a guarded agent loop — Python/NumPy, no RAG frameworks. Azure adapters (AI Search, OpenAI, Graph, Teams) documented and mock-tested.`
- **Topics**: `rag` `retrieval-augmented-generation` `information-retrieval`
  `bm25` `hybrid-search` `rrf` `fastapi` `agents` `tool-calling` `numpy`
  `prompt-engineering` `evaluation` `azure` `copilot-studio` `python`
- **README header block**: keep the current README.md header; optionally add
  your measured eval table (`eval_out/results.md` converted to an image or
  pasted rows) under a heading like "Measured results (my laptop, <date>)" —
  ONLY after running the real eval.

## 30-second spoken pitch

"This is a RAG knowledge system I built from scratch — no LangChain, just
Python and NumPy. It ingests policy documents with per-document access
roles, chunks and embeds them, and answers with hybrid search: dense cosine
plus hand-implemented BM25 fused with Reciprocal Rank Fusion. Every answer
carries validated citations; questions it cannot support get refused
before the model is even called. There's a FastAPI API with streaming, a
metrics store with hashed questions and SQL analytics, and a hand-written
agent loop that can open a CRM ticket and post to Teams when the knowledge
base has no answer. Everything is tested offline with fakes — two hundred
plus tests — and the Azure equivalents are documented as swappable
adapters."

## 2-minute spoken pitch

Add to the 30-second version: the security story (ACL applied before
ranking so a restricted chunk can never even be a candidate; PII redaction
in the logs; injection phrase heuristics with a canary test in the eval
set), the evaluation story (50-item set in four categories — in-scope,
out-of-scope, injection, access-control — validated in CI; a sweep over
chunk sizes and retrieval modes; three prompt versions A/B-tested on the
same questions), and the honest Azure story (I know what maps to what —
my vector store to AI Search, my RRF to its hybrid mode, my API key to Entra
ID — but the adapters are mock-tested and labelled NOT TESTED AGAINST LIVE
SERVICE, because I have not worked in a live tenant yet). Close with: "the
point of the project is that I can explain every line, because I wrote
every line."

## Reply email to the recruiter (blanks are YOURS — never invent)

> Subject: Re: AI Solution Engineer – Enterprise Automation & Copilot
> Systems – interest confirmed
>
> Dear [recruiter's name],
>
> Thank you for reaching out. I am interested in the AI Solution Engineer
> (Enterprise Automation & Copilot Systems) position, and I am confirming
> the details you asked for below.
>
> - Current/last job role: [your role, employer, dates]
> - Reason for change: [one honest sentence, e.g. "seeking applied AI
>   engineering work closer to production systems"]
> - Current location: [city]
> - Salary details: current CTC [___], in-hand [___]
> - Expected salary: [___]
> - Notice period: [___]
> - Availability for interview: [dates/time windows]
> - Total experience: [___]
>
> As supporting material, my most recent project is a RAG knowledge system
> built from scratch in Python (hybrid retrieval, access control, a
> FastAPI API and a tool-calling agent), with a documented mapping to the
> Azure stack used in the role — the repository link is in my signature
> and I am happy to walk through any part of it in an interview.
>
> Best regards,
> [your name]
> [phone] · [email] · [GitHub profile URL]

## Numbers I must measure before putting them on my resume

Each command below prints or writes the number; run on the hardware you
will mention, then paste the value into the matching bullet. All values
are approximate single-run measurements — say so.

| Number | Exact command that produces it |
|---|---|
| Test count (offline) | `pytest -q` (read the "N passed, M deselected" line) |
| recall@4 / MRR / nDCG for your chosen config | `python rag/evaluation/run_eval.py --provider ollama` then read `eval_out/results.csv` |
| p50 / p95 latency, TTFT (ms) | same file, `latency_p50_ms`, `latency_p95_ms`, `ttft_p50_ms` columns |
| Prompt v1 -> v3 improvement (groundedness, refusal, injection, citations) | `python rag/evaluation/prompt_ab.py --provider ollama` then read `eval_out/prompt_ab.md` |
| No-answer rate on out-of-scope questions | `sqlite3 logs/metrics.db` then query 2 in `sql/analytics.sql` after a real eval run |
| Access-control leak rate | must be `0.0`; verify with `python rag/evaluation/run_eval.py --provider ollama` (leak_rate column) — assert it, don't boast it |
| Corpus/eval size sanity | `python scripts/validate_eval_set.py` (50 items, counts printed) |
| IVF recall/latency table | `python scripts/benchmark.py --n 5000` (your machine) |

After a clean install on the target machine, also run
`pip freeze > requirements.lock.txt` and commit it, so anyone reproducing
your numbers gets your exact dependency versions.
