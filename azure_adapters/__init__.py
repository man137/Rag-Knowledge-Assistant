"""azure_adapters — cloud bridges behind the same local interfaces.

WHAT: package marker for the Azure-side adapters: azure_openai_llm.py and
azure_openai_embedder.py (Azure OpenAI), azure_ai_search_retriever.py (Azure
AI Search hybrid retrieval behind the rag.hybrid Retriever interface),
graph_sharepoint_loader.py (SharePoint ingestion via Microsoft Graph + msal)
and teams_webhook.py (Teams incoming webhook).
WHY: the local repo implements every interface by hand; these adapters are
the swappable cloud equivalents — same Protocols, lazy SDK imports, tested
with MOCKED clients.
MATHS: none.
FAILURE MODE: every adapter here is NOT TESTED AGAINST LIVE SERVICE — they
are verified in the offline test-suite with injected mock clients only.
Cloud SDK imports are lazy so the rest of the repo (and `pytest -q`) runs
without the SDKs installed.
INTERVIEW ONE-LINER: "The cloud adapters implement the same Protocols as my
hand-written core, so switching from my NumPy store to Azure AI Search is a
constructor argument, not a rewrite."
"""

__version__ = "1.0.0"
