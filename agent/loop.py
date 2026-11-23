"""The plan-act-observe agent loop, hand-written (no framework).

WHAT: AgentLoop.run(goal) drives the LLM through JSON tool-calls: build the
prompt (system instructions + tool schemas + goal + observations), ask for
ONE JSON action, validate it (allowlist via the registry, one repair attempt
for malformed JSON), execute, append the observation, repeat until
"final_answer" or the step limit. Every thought/action/observation is logged
with timestamps and kept in a trace list returned to the caller.
WHY: the JD asks for "AI agent workflows"; writing the loop by hand is the
whole point of this layer — the same mechanics Copilot Studio generates
declaratively are visible here line by line.
MATHS: none (control flow). The step limit is the safety bound: a loop that
never reaches final_answer terminates deterministically.
FAILURE MODE: a model that emits prose instead of JSON gets exactly ONE
repair attempt per run (the JSON error is fed back as an observation and the
retry counts as its own step); if that second reply is still invalid the
loop stops with ok=False. A tool error is an observation, not a crash — the
model can switch strategy. A tool not on the allowlist is never executed.
INTERVIEW ONE-LINER: "My agent loop is a few hundred lines: prompt -> JSON
action -> allowlist + schema validation -> observation, with one JSON repair
attempt, a step limit and a full timestamped trace."
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agent.tools import ALLOWED_TOOLS, TOOL_SCHEMAS, ToolRegistry

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a support agent for Aurora Atlas Events, a synthetic
events and travel company.

TOOLS:
{tool_descriptions}

PROTOCOL - reply with ONE JSON object and nothing else:
{{"thought": "why you chose this action",
  "action": "<tool name from the list, or final_answer>",
  "args": {{...}},
  "answer": "<only present when action is final_answer>"}}

Write tools (create_crm_ticket, post_teams_message) need an extra
argument "confirm": true. Use them only when the knowledge base does not
answer the question or the situation matches an escalation rule.
When the KB answers the question, keep the cited chunk ids in your
final answer as [source:chunk_id].
"""

_FINAL_ACTION = "final_answer"


@dataclass
class AgentResult:
    """Everything one agent run produced."""

    goal: str
    final_answer: str
    ok: bool                     # reached final_answer within the step limit
    steps_used: int
    trace: list[dict[str, Any]] = field(default_factory=list)


def _tool_descriptions() -> str:
    lines = []
    for name in ALLOWED_TOOLS:
        spec = TOOL_SCHEMAS[name]
        args = ", ".join(spec.get("required", []))
        lines.append(f"- {name}({args}): {spec['description']}")
    return "\n".join(lines)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


class AgentLoop:
    """Plan-act-observe loop over a ToolRegistry with a step limit."""

    def __init__(self, registry: ToolRegistry, llm, max_steps: int = 8) -> None:
        self.registry = registry
        self.llm = llm
        self.max_steps = max_steps

    # ------------------------------------------------------------------ #
    def _parse_action(self, raw: str) -> dict[str, Any]:
        """Parse the model reply as a single JSON action object. Tolerates
        surrounding prose by extracting the first {...} block."""
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def run(self, goal: str) -> AgentResult:
        messages: list[dict[str, str]] = [
            {"role": "system",
             "content": SYSTEM_PROMPT.format(
                 tool_descriptions=_tool_descriptions())},
            {"role": "user", "content": f"GOAL: {goal}"},
        ]
        trace: list[dict[str, Any]] = []
        step = 0
        repair_used = False
        while step < self.max_steps:
            step += 1
            entry: dict[str, Any] = {"step": step, "ts": _now()}
            raw = str(self.llm.generate(messages, stream=False))
            messages.append({"role": "assistant", "content": raw})

            # ---- parse (with exactly ONE repair attempt per run) ----
            try:
                action_obj = self._parse_action(raw)
            except (json.JSONDecodeError, ValueError):
                entry["error"] = ("invalid JSON after repair attempt"
                                  if repair_used else "invalid JSON")
                entry["raw"] = raw[:200]
                trace.append(entry)
                if repair_used:
                    return AgentResult(
                        goal=goal,
                        final_answer=("I could not produce a valid action "
                                      "plan for this request."),
                        ok=False, steps_used=step, trace=trace)
                repair_used = True
                messages.append({
                    "role": "user",
                    "content": ("OBSERVATION: your reply was not valid JSON "
                                 f"({raw[:120]!r}). Reply again with ONE "
                                 "JSON object following the protocol."),
                })
                continue   # the repair is its own step

            thought = str(action_obj.get("thought", ""))
            action = str(action_obj.get("action", ""))
            args = action_obj.get("args", {}) or {}
            entry.update({"thought": thought, "action": action, "args": args})
            log.info("agent step %d action=%s thought=%s", step, action,
                     thought[:80])

            # ---- final answer terminates ----
            if action == _FINAL_ACTION:
                answer = str(action_obj.get("answer", "")).strip() or \
                    "I could not find an answer."
                entry["final_answer"] = answer
                trace.append(entry)
                return AgentResult(goal=goal, final_answer=answer, ok=True,
                                   steps_used=step, trace=trace)

            # ---- execute the tool (allowlist + schema + confirm inside) ----
            observation = self.registry.call(action, args)
            entry["observation"] = observation
            entry["ok"] = observation.get("ok", False)
            trace.append(entry)
            messages.append({
                "role": "user",
                "content": f"OBSERVATION: {json.dumps(observation, ensure_ascii=False)}",
            })

        # ---- step limit reached without a final answer ----
        return AgentResult(
            goal=goal,
            final_answer=(f"I could not complete this task within "
                          f"{self.max_steps} steps."),
            ok=False, steps_used=self.max_steps, trace=trace)
