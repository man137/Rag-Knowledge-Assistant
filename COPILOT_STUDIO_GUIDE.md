# COPILOT_STUDIO_GUIDE — connect this project to Copilot Studio / Power Automate

This guide walks a student with a **free or trial tenant** from the FastAPI
service to a Copilot Studio topic that answers from the hand-built RAG
pipeline. Everything UI-specific below can drift; steps marked
**[verify in current docs]** should be checked against Microsoft Learn for the
tenant you are using — Copilot Studio's navigation changes frequently.

What this guide covers but the repo does NOT include: running code inside a
tenant. No screenshots, no saved connector definitions — those must be
produced by you, in your own tenant, with your own (free) resources.

---

## 0. Prerequisites

- A Microsoft 365 Developer Program tenant (free) or a work/school tenant
  with a Copilot Studio trial. **[verify in current docs]**: trial signup
  steps at learn.microsoft.com/copilot-studio/fundamentals-what-is.
- The API running over HTTPS (step 1) and your `openapi.json`
  (`make openapi` generates it at the repo root).

## 1. Expose the FastAPI service over HTTPS

Copilot Studio custom connectors require a public HTTPS endpoint with a
valid certificate. Pick one:

- **dev tunnel (quickest)**: run `uvicorn api.main:app --port 8000` locally
  and expose it with a tunnel tool that gives you a public HTTPS URL.
- **free cloud host**: deploy the `Dockerfile` image to any container host
  that terminates TLS for you.

Set `API_KEY=<strong random value>` in the service environment — do not
leave auth disabled when the endpoint is public. Keep
`CORS_ORIGINS` aligned with any browser-based client you add.

Smoke-test it:
```bash
curl https://<your-host>/health
curl -N -X POST https://<your-host>/ask \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question":"What is the refund rule for event cancellation?"}'
```

## 2. Import the custom connector

1. Generate the contract: `make openapi` (writes `openapi.json`; the test
   suite verifies it contains `/ask`, `/ask_sync`, `/feedback`, `/metrics`).
2. In Power Platform: **Custom connectors > New custom connector > Import
   OpenAPI file** and upload `openapi.json`. **[verify in current docs]**
   — the exact menu path changes between Power Apps and Copilot Studio.
3. **Security tab**: choose **API Key** authentication with the header
   **`X-API-Key`** (the schema in `api/schemas.py` matches what the wizard
   expects). Paste your key in the connection when creating it.
4. **Definition tab**: check that `/ask_sync` and `/ask` appear with their
   request bodies; rename actions to something friendly
   ("Ask knowledge base", "Ask and stream").
5. **Test tab**: create a connection and send
   `{"question":"What is the hotel cap in metro cities?","mode":"hybrid","k":4}`
   to `/ask_sync`. You should get JSON with `answer`, `sources` and
   `no_answer:false`.

Note on limits: connectors must respond within their timeout window. The
pinned local Ollama stack answers quickly on small prompts; if you host a
slower model, prefer a smaller `k` and shorter chunks (`CHUNK_SIZE`).

## 3. Call /ask from Copilot Studio or Power Automate

**SSE limitation (why /ask_sync exists)**: `/ask` streams Server-Sent Events,
which custom connector actions and Power Automate cannot consume as a
stream. The repo ships `POST /ask_sync` — the identical pipeline without
streaming — precisely for this integration, and its tests assert it returns
the full response JSON.

- **Copilot Studio topic**: create a topic, add a **Call an action** node
  bound to your connector's `/ask_sync` action, map the user's question
  `Topic.Question` into the `question` field, and set `X-User-Role` from a
  topic variable (e.g. "staff"). Return `{answer}` to the user, then list
  `{sources[0].title}`-style citations in a follow-up message.
  **[verify in current docs]** — action nodes and variables have per-release
  UI differences.
- **Power Automate flow**: the same connector action works in a flow: trigger
  (e.g. "when a new item is created" or a Teams trigger) -> **Ask knowledge
  base** (`/ask_sync`) -> post the answer somewhere (step 5).

Role handling matters: pass the real user's role so the ACL actually
protects content. In a production tenant that role comes from Entra ID
claims, not from a hand-set variable — the demo header is explicitly
documented as a stand-in (`api/auth.py`).

## 4. Alternative: Copilot Studio generative answers over SharePoint

Without any custom code, Copilot Studio can ground answers in a SharePoint
site:

1. Upload `data/docs/*.md` to a document library in the tenant.
2. In Copilot Studio, add a knowledge source (SharePoint site/library).
   **[verify in current docs]** — knowledge source setup differs between
   "generative answers" configurations.
3. Ask the same eval questions inside the Copilot.

This path costs no code but gives you less control: chunking, hybrid
retrieval, the no-answer threshold, citation validation and the role ACL are
Copilot Studio's, not yours — which is exactly why step 6 compares the two.

## 5. Post the result to Teams from a flow

1. In your flow, after the **Ask knowledge base** action, add a **Post
   message in a chat or channel** (Teams) action.
   **[verify in current docs]** — the Teams connector's action names change.
2. Compose the message from the action outputs: the `answer` field plus the
   top source title, e.g.
   `@{outputs('Ask_knowledge_base')?['body/answer']} (source: @{outputs('Ask_knowledge_base')?['body/sources'][0]['title']})`.
3. Escalation variant: when `no_answer` is true, post to `#support-escalations`
   instead of answering — the flow version of `scripts/run_agent_demo.py`,
   which does the same with the hand-written agent loop.

If you use the classic incoming-webhook path instead of the Teams connector,
`azure_adapters/teams_webhook.py` is the POST the flow would make —
NOT TESTED AGAINST LIVE SERVICE.

## 6. Compare answer quality: custom RAG vs Copilot Studio native

Run the same eval set through both systems and record the numbers yourself:

1. Run the repo side for real (needs a real model, not `--fake`):
   ```bash
   python rag/evaluation/run_eval.py --provider ollama   # writes eval_out/results.csv
   ```
2. Ask the 30 in-scope questions from `data/eval/eval_set.jsonl` inside your
   Copilot Studio agent (type them in; there is no batch API for manual
   testing) and mark for each: did it answer, did it cite, would it have
   refused the HR question for a guest?
3. Compare on: refusal behaviour on the out-of-scope and access-control
   items, citation quality, and (subjectively) consistency across rephrasings.
4. Keep the comparison honest: note the model, the date, and that your
   numbers come from a 50-item synthetic set.

**Expected outcome (a hypothesis to verify, not a fact)**: the managed
pipeline is smoother to stand up; the custom one gives you measurable
control — your own chunking, thresholds, evals and ACLs — and a debugging
trace for every answer.
