# Azure bridge — what maps to what, and what stays a mock

Every adapter in `azure_adapters/` carries the label **NOT TESTED AGAINST
LIVE SERVICE**: they are exercised only against injected mock clients in
`tests/test_azure_adapters.py`. Nothing in this project has been deployed to
or run against a real Azure tenant.

## Hand-built component -> Azure equivalent

| Hand-built (this repo) | Azure equivalent | What changes in production |
|---|---|---|
| `rag/vector_store.py` (NumPy matrix, argpartition top-k, IVF) | Azure AI Search vector index | ANN search, persistence and replicas are managed; delete `save()/load()` |
| `rag/bm25.py` (Okapi BM25, inverted index) | Azure AI Search BM25 half of hybrid search | Managed BM25; per-field analyzers instead of my regex tokenizer |
| `rag/hybrid.py` (RRF, k=60) | Azure AI Search hybrid retrieval | The service fuses BM25 + vector internally; scores are on a different scale, so thresholds re-calibrate |
| `rag/reranker.py` (cross-encoder) | Semantic ranker / semantic search | The managed ranker replaces the local cross-encoder; keep the two-stage retrieve-then-rerank shape |
| `rag/embedder.py` (SentenceTransformers) | Azure OpenAI embeddings deployment | Vectors must come from ONE model for index and queries; re-index after any change |
| `rag/generator.py` -> Ollama / OpenAI-compat | Azure OpenAI chat deployment | Deployment name replaces model name; content filters + quotas apply |
| `azure_adapters/azure_ai_search_retriever.py` | Azure AI Search SDK retriever | This IS the swap: same Retriever interface, OData ACL filter instead of the NumPy mask |
| `azure_adapters/graph_sharepoint_loader.py` | Microsoft Graph + SharePoint ingestion | Entra app registration with `Sites.Read.All` admin consent; msal client-credentials |
| `azure_adapters/teams_webhook.py` | Teams incoming webhook / Workflows | Classic webhooks take `{"text": ...}`; Workflows take an adaptive card payload |
| `api/main.py` (FastAPI, uvicorn) | Azure Container Apps / App Service | Managed TLS, scaling and front-door; keep the container as built by the Dockerfile |
| `observability/logging_setup.py` (JSON logs) | Application Insights | Same JSON fields become telemetry; request_id becomes the operation id |
| `observability/metrics_store.py` (SQLite query_log) | Azure SQL / Log Analytics | The 6 analytics queries move to KQL/SQL; keep hashed questions |
| `api/auth.py` (API key, constant time) | Microsoft Entra ID (OAuth 2.0) | Bearer token validated against the tenant; roles from token CLAIMS, never from a header |
| `.env` secrets | Azure Key Vault | Secrets fetched at startup by managed identity; nothing in images or env files |
| `agent/crm_mock.py` (SQLite tickets) | Dynamics 365 CRM connector | Real ticket ids, statuses and the client's SLA rules |
| `agent/teams_mock.py` | Teams webhook adapter above | Same `post(channel, text)` signature |

## Adapter-by-adapter status

| File | Import style | Tested how |
|---|---|---|
| `azure_openai_llm.py` | lazy `from openai import AzureOpenAI` | mocked client factory |
| `azure_openai_embedder.py` | lazy `from openai import AzureOpenAI` | mocked client factory |
| `azure_ai_search_retriever.py` | lazy `from azure.search.documents ...` | mocked SearchClient |
| `graph_sharepoint_loader.py` | lazy `import msal` | mocked HTTP + token provider |
| `teams_webhook.py` | httpx only | httpx.MockTransport |

The `openai`, `azure-search-documents` and `msal` packages are pinned in
`requirements.txt` but imported ONLY inside these adapters, so `pytest -q`
passes without any of them installed.

## What would change in production (honest list)

1. **Auth**: replace the demo API key with Entra ID bearer tokens; roles must
   come from validated token claims, not the `X-User-Role` header.
2. **Secrets**: Key Vault + managed identity; no `.env` anywhere.
3. **Retrieval**: swap the NumPy store for the AI Search retriever — the
   pipeline only depends on the `Retriever` protocol, so this is one
   constructor. Re-calibrate `no_answer_threshold`: Azure's fused scores are
   not on the cosine scale.
4. **Ingestion**: run the Graph/SharePoint loader on a schedule (Container
   Apps job or Logic App), delete-and-rebuild per source to avoid stale
   indexes.
5. **Observability**: ship the JSON logs to Application Insights; replace the
   SQLite metrics table with a managed store and translate
   `sql/analytics.sql` to KQL.
6. **Agent tools**: point `create_crm_ticket` at Dynamics 365 and
   `post_teams_message` at the real webhook; keep the `confirm=true` gate.
7. **Evaluation**: re-run `run_eval.py` against the Azure retrieval path with
   the same eval set before claiming any quality number.

## Version-sensitive calls (each marked `# VERSION-SENSITIVE:` in code)

- `AzureOpenAI(...)` client kwargs (openai >= 1.0).
- `client.chat.completions.create(model=<deployment name>, ...)`.
- `client.embeddings.create(model=<deployment name>, input=[...])`.
- `SearchClient(...)` and `VectorizedQuery(...)` (azure-search-documents
  >= 11.4); the adapter falls back to the older `vector=` kwarg.
- Graph `v1.0` REST paths (`/sites/{id}/drive/root:/...:/children`).
- Teams classic-webhook payload vs Workflows adaptive-card payload.
- `msal.ConfidentialClientApplication` + `acquire_token_for_app`.
