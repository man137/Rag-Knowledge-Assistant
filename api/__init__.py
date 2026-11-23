"""api — FastAPI service exposing the pipeline (SSE) plus ops endpoints.

WHAT: package marker for the FastAPI app: schemas.py (request models),
auth.py (API key + role header) and main.py (endpoints, SSE streaming,
ingest, feedback, metrics, health).
WHY: an HTTP API is how a RAG system is actually consumed (the UI, Copilot
Studio custom connectors, Power Automate flows all need it).
MATHS: none.
FAILURE MODE: the API never returns stack traces (global handler) and never
stores raw questions (metrics rows are hashed by the pipeline).
INTERVIEW ONE-LINER: "One FastAPI service with SSE streaming answers, a
non-streaming /ask_sync for Power Automate, and API-key auth designed to be
swapped for Entra ID claims."
"""

__version__ = "1.0.0"
