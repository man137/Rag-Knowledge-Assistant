"""Tool registry: JSON schemas, allowlist, confirm gating, validation.

WHAT: a registry of agent tools — search_kb(query), create_crm_ticket(summary,
priority), post_teams_message(channel, text), lookup_ticket(id) — each with
a JSON schema. `call` enforces, in order: allowlist membership, schema
validation (required keys, types, enums, no unknown keys), and the write
guard (create_crm_ticket / post_teams_message require args["confirm"] ==
true or the call is refused). Errors come back as observation dicts, never
exceptions.
WHY: an agent that can call anything is a security incident waiting to
happen; the allowlist plus per-call argument validation is the minimum
enterprise guard-railing, and confirm=true is the human-in-the-loop gate for
write actions.
MATHS: none.
FAILURE MODE: schema validation is deliberately simple (type + enum +
required + unknown-key checks, no JSON-Schema-compliant parser) — nested
schemas or $ref would need a real validator. Unknown extra arguments are
REJECTED, not ignored, so a confused model cannot smuggle options in.
INTERVIEW ONE-LINER: "Every tool call passes an allowlist, a schema check
and — for writes — an explicit confirm=true gate, and failures come back as
observations the model can react to."
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from agent.crm_mock import PRIORITIES, CrmMock
from agent.teams_mock import TeamsMock

log = logging.getLogger(__name__)

# Minimal JSON-schema-like descriptors. "write": true -> requires confirm.
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_kb": {
        "description": "Search the synthetic events/travel knowledge base "
                       "and return a grounded answer with citations.",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "write": False,
    },
    "lookup_ticket": {
        "description": "Look up a CRM ticket by its numeric id.",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
        "write": False,
    },
    "create_crm_ticket": {
        "description": "Create a CRM ticket. Write action: requires "
                       "confirm=true.",
        "properties": {
            "summary": {"type": "string"},
            "priority": {"type": "string", "enum": list(PRIORITIES)},
            "confirm": {"type": "boolean"},
        },
        "required": ["summary", "priority", "confirm"],
        "write": True,
    },
    "post_teams_message": {
        "description": "Post a message to a Teams channel. Write action: "
                       "requires confirm=true.",
        "properties": {
            "channel": {"type": "string"},
            "text": {"type": "string"},
            "confirm": {"type": "boolean"},
        },
        "required": ["channel", "text", "confirm"],
        "write": True,
    },
}

ALLOWED_TOOLS = tuple(TOOL_SCHEMAS)


class ToolRegistry:
    """Allowlist + schema validation + confirm gating around tool functions."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._fns: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any],
                 schema: dict[str, Any] | None = None) -> None:
        if name in self._tools:
            raise ValueError(f"tool {name!r} already registered")
        self._tools[name] = schema if schema is not None \
            else dict(TOOL_SCHEMAS.get(name, {"properties": {}, "required": [],
                                               "write": False}))
        self._fns[name] = fn

    def allowed(self) -> list[str]:
        return sorted(self._tools)

    # ---------------------------------------------------------------- #
    @staticmethod
    def _validate(name: str, schema: dict[str, Any],
                  args: dict[str, Any]) -> str | None:
        """Return an error string, or None when the arguments are valid.
        The write guard runs FIRST so a refused write always explains the
        confirm requirement, whatever else is wrong with the arguments."""
        if not isinstance(args, dict):
            return f"{name}: args must be a JSON object"
        if schema.get("write") and args.get("confirm") is not True:
            return (f"{name}: this is a write action and requires "
                    "confirm=true in the arguments")
        properties = schema.get("properties", {})
        for key in args:
            if key not in properties:
                return f"{name}: unknown argument {key!r}"
        for key in schema.get("required", []):
            if key not in args:
                return f"{name}: missing required argument {key!r}"
        for key, value in args.items():
            spec = properties[key]
            expected = spec.get("type")
            if expected == "string" and not isinstance(value, str):
                return f"{name}: argument {key!r} must be a string"
            if expected == "integer" and (not isinstance(value, int)
                                          or isinstance(value, bool)):
                return f"{name}: argument {key!r} must be an integer"
            if expected == "boolean" and not isinstance(value, bool):
                return f"{name}: argument {key!r} must be a boolean"
            if "enum" in spec and value not in spec["enum"]:
                return (f"{name}: argument {key!r} must be one of "
                        f"{spec['enum']}, got {value!r}")
            if expected == "string" and not str(value).strip():
                return f"{name}: argument {key!r} must not be empty"
        return None

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool call. Returns {"ok": True, "result": ...} or
        {"ok": False, "error": ...} — errors are observations, not exceptions."""
        if name not in self._tools:
            # NEVER execute a tool that is not on the allowlist
            return {"ok": False,
                    "error": f"tool {name!r} is not on the allowlist; "
                             f"allowed tools: {sorted(self._tools)}"}
        schema = self._tools[name]
        if not isinstance(args, dict):
            return {"ok": False, "error": f"{name}: args must be a JSON object"}
        error = self._validate(name, schema, args)
        if error:
            return {"ok": False, "error": error}
        try:
            result = self._fns[name](**args)
        except Exception as exc:  # noqa: BLE001 - tool errors are observations
            log.warning("tool %s raised %s: %s", name, type(exc).__name__, exc)
            return {"ok": False,
                    "error": f"{name} failed: {type(exc).__name__}: {exc}"}
        return {"ok": True, "result": result}


def build_default_registry(pipeline, crm: CrmMock, teams: TeamsMock,
                           user_role: str = "guest") -> ToolRegistry:
    """Wire the four standard tools around a RAG pipeline and the mocks."""
    registry = ToolRegistry()

    def search_kb(query: str) -> dict[str, Any]:
        response = pipeline.ask(query, mode="hybrid", user_role=user_role)
        sources = [{"chunk_id": s.chunk_id, "source": s.source,
                    "title": s.title, "score": s.score}
                   for s in response.sources]
        return {"answer": response.answer, "no_answer": response.no_answer,
                "sources": sources,
                "citations": response.valid_citations}

    def lookup_ticket(id: int) -> Any:  # noqa: A002 - schema arg name
        ticket = crm.get_ticket(id)
        if ticket is None:
            # raise, not return: the registry turns exceptions into
            # {"ok": False, "error": ...} observations for the loop
            raise ValueError(f"no ticket with id {id}")
        return ticket

    registry.register("search_kb", search_kb, TOOL_SCHEMAS["search_kb"])
    registry.register("lookup_ticket", lookup_ticket,
                      TOOL_SCHEMAS["lookup_ticket"])
    registry.register(
        "create_crm_ticket",
        lambda summary, priority, confirm: crm.create_ticket(summary, priority),
        TOOL_SCHEMAS["create_crm_ticket"])
    registry.register(
        "post_teams_message",
        lambda channel, text, confirm: teams.post(channel, text),
        TOOL_SCHEMAS["post_teams_message"])
    return registry
