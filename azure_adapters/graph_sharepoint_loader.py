"""SharePoint document library loader via Microsoft Graph + msal.

WHAT: `GraphSharePointLoader.list_files()` and `.load()` list a folder in a
SharePoint document library through the Graph API and download its text
files, returning rag.loaders.Document objects whose `allowed_roles` metadata
comes from a configurable path->roles mapping — the ACL survives ingestion
from the cloud exactly as it does locally.
WHY: "enterprise tool integration" in the JD means pulling knowledge from
where it actually lives (SharePoint); the front-matter roles of the local
corpus map to the same field, so the retrieval ACL does not change.
MATHS: none.
FAILURE MODE: msal's ConfidentialClientApplication + acquire_token_for_app
(client credentials flow) needs an Entra ID app registration with the
Sites.Read.All / Files.Read.All application permission granted admin-consent;
missing consent returns an interaction_required error, not an exception you
can retry. Graph API paths can change between versions — the URL construction
below is pinned to v1.0 and marked VERSION-SENSITIVE.
NOT TESTED AGAINST LIVE SERVICE: exercised only with a mocked HTTP layer and
token provider in tests/test_azure_adapters.py.
INTERVIEW ONE-LINER: "Ingestion from SharePoint is the same Document objects
with allowed_roles from a path mapping — the security model travels with the
content from cloud to index."
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from rag.loaders import Document

log = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"


class GraphSharePointLoader:
    """Lists and downloads SharePoint files via Graph. NOT TESTED AGAINST
    LIVE SERVICE (mocked HTTP + token in tests only)."""

    def __init__(self, tenant_id: str, client_id: str, client_secret: str,
                 site_id: str, folder_path: str,
                 role_map: dict[str, list[str]] | None = None,
                 default_roles: list[str] | None = None,
                 http: Any = None,
                 token_provider: Callable[[], str] | None = None) -> None:
        if not (tenant_id and client_id and client_secret and site_id):
            raise ValueError(
                "Graph loader needs GRAPH_TENANT_ID, GRAPH_CLIENT_ID, "
                "GRAPH_CLIENT_SECRET and SHAREPOINT_SITE_ID.")
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.site_id = site_id
        self.folder_path = folder_path.strip("/")
        # path substring -> roles; anything unmatched gets default_roles
        self.role_map = dict(role_map or {})
        self.default_roles = list(default_roles or ["all"])
        self._http = http                      # injectable httpx-like client
        self._token_provider = token_provider  # injectable for tests

    # ------------------------------------------------------------------ #
    def _token(self) -> str:
        if self._token_provider is not None:
            return self._token_provider()
        try:
            # VERSION-SENSITIVE: msal ConfidentialClientApplication with the
            # client-credentials flow. scopes must be the Graph default.
            import msal
        except ImportError as exc:
            raise RuntimeError(
                "The 'msal' package is not installed; it is required only "
                "for the SharePoint loader. Install with "
                "'pip install -r requirements.txt'."
            ) from exc
        app = msal.ConfidentialClientApplication(
            self.client_id,
            client_credential=self.client_secret,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}",
        )
        result = app.acquire_token_for_app(
            scopes=["https://graph.microsoft.com/.default"])
        if "access_token" not in result:
            raise RuntimeError(
                f"Graph token acquisition failed: {result.get('error')}: "
                f"{result.get('error_description', '')[:200]}")
        return result["access_token"]

    def _get(self, url: str) -> Any:
        headers = {"Authorization": f"Bearer {self._token()}"}
        if self._http is not None:
            response = self._http.get(url, headers=headers)
        else:
            import httpx

            response = httpx.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response

    def _roles_for(self, name: str) -> list[str]:
        lowered = name.lower()
        for pattern, roles in self.role_map.items():
            if pattern.lower() in lowered:
                return list(roles)
        return list(self.default_roles)

    # ------------------------------------------------------------------ #
    def list_files(self) -> list[dict[str, Any]]:
        """One page of drive items under the configured folder path.
        VERSION-SENSITIVE: pinned to Graph v1.0 drive-root child listing."""
        url = (f"{GRAPH_BASE}/sites/{self.site_id}/drive/root:"
               f"/{self.folder_path}:/children")
        response = self._get(url)
        payload = response.json()
        return [item for item in payload.get("value", [])
                if item.get("file") is not None]

    def load(self, suffixes: tuple[str, ...] = (".md", ".txt")) -> list[Document]:
        """Download every listed text file and wrap it as a Document with
        allowed_roles from the role map."""
        documents: list[Document] = []
        for item in self.list_files():
            name = item.get("name", "")
            if not name.lower().endswith(suffixes):
                continue
            url = (f"{GRAPH_BASE}/sites/{self.site_id}/drive/items/"
                   f"{item['id']}/content")
            response = self._get(url)
            documents.append(Document(
                text=response.text,
                source=f"sharepoint:{self.folder_path}/{name}",
                page=1,
                metadata={
                    "title": name,
                    "source": name,
                    "allowed_roles": self._roles_for(name),
                    "web_url": item.get("webUrl", ""),
                }))
        log.info("Graph loader: %d documents from %s", len(documents),
                 self.folder_path)
        return documents


def role_map_from_env(raw: str) -> dict[str, list[str]]:
    """Parse SHAREPOINT_ROLE_MAP="policy:all,staff;hr:hr" into a dict.
    Split on ';', then ':', roles comma-separated."""
    mapping: dict[str, list[str]] = {}
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        pattern, _, roles = chunk.partition(":")
        mapping[pattern.strip()] = [r.strip().lower()
                                    for r in roles.split(",") if r.strip()]
    return mapping
