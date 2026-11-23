"""API-key authentication and role resolution for the demo service.

WHAT: `verify_api_key` compares the X-API-Key header against settings.api_key
in CONSTANT TIME (hmac.compare_digest) and `role_from_header` maps the
X-User-Role header to a known role, defaulting unknown/missing roles to
guest (least privilege).
WHY: the demo needs an auth boundary that is simple to demo and safe to
replace; constant-time comparison prevents the classic timing side channel
on key checks.
MATHS: none.
FAILURE MODE: AN HONEST LIMIT — API keys in a single header are a DEMO
mechanism. Production for this service would use Microsoft Entra ID: the
caller presents an OAuth 2.0 bearer token, the API validates it against the
tenant, and the ROLE comes from token claims (groups / app roles), not from
a client-supplied header. When settings.api_key is EMPTY, auth is DISABLED
(local development only) and the app logs a warning at startup. The role
header is likewise client-supplied for the demo: in production the role must
come from the validated token, never from the request.
INTERVIEW ONE-LINER: "Demo auth is a constant-time API key check; the swap
to production is Entra ID bearer tokens with roles read from claims — I
marked exactly where that change happens."
"""

from __future__ import annotations

import hmac
import logging

from rag.security import normalise_role

log = logging.getLogger(__name__)


def verify_api_key(provided: str | None, expected: str) -> bool:
    """True when the key matches. An EMPTY expected key disables auth (dev).

    hmac.compare_digest runs in time independent of where the strings
    differ, so an attacker cannot leak the key byte-by-byte via response
    timing."""
    if not expected:
        # auth intentionally disabled for local development
        return True
    return hmac.compare_digest(str(provided or ""), expected)


def role_from_header(role: str | None) -> str:
    """Normalise the X-User-Role header; unknown roles degrade to guest."""
    return normalise_role(role)
