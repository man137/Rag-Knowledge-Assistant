#!/usr/bin/env python3
"""Agent demo: the event-cancellation refund use case, end to end.

WHAT: runs the JD's enterprise-agent scenario against the hand-written
plan-act-observe loop: "A customer asks about the refund rule for an event
cancellation. The agent answers from the KB with citations; if the answer is
not in the KB, it creates a CRM ticket and posts a Teams message to
#support-escalations." Two scenarios run back to back: an in-scope question
(answered with citations) and an out-of-KB question (ticket + Teams
escalation). Every thought/action/observation is printed with timestamps.
By default a SCRIPTED FakeLLM drives the loop deterministically (a labelled
mock, see DemoFakeLLM below); --live switches to the provider configured in
the environment, where the real model decides the actions.
WHY: this script is the 2-minute demo of agent workflows + enterprise tool
integration, runnable offline with zero credentials.
MATHS: none.
FAILURE MODE: the scripted fake follows the happy paths only; it exists to
demonstrate LOOP MECHANICS, not model intelligence. With --live the trace
depends entirely on the configured model's ability to follow the JSON
protocol (the loop's one-repair and allowlist rules bound the damage).
INTERVIEW ONE-LINER: "One script demos both agent paths — answer-with-
citations from the KB, and escalate-via-tools when the KB has nothing —
with a full timestamped trace and the Teams payload written to a log."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.crm_mock import CrmMock  # noqa: E402
from agent.loop import AgentLoop  # noqa: E402
from agent.teams_mock import TeamsMock  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402
from api.main import load_or_build_pipeline  # noqa: E402
from observability.logging_setup import configure_logging  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.generator import FakeLLM, get_provider  # noqa: E402

SCENARIO_IN_SCOPE = ("A customer asks about the refund rule for an event "
                     "cancellation: how much do they get back if they cancel "
                     "20 days before the event?")
# deliberately shares almost no vocabulary with the corpus so retrieval
# confidence stays below the no-answer threshold and the escalation path runs
SCENARIO_OUT_OF_KB = ("A customer asks whether we accept cryptocurrency "
                      "payments for NFT gallery passes.")


class DemoFakeLLM(FakeLLM):
    """A SCRIPTED fake: deterministic decisions based on the last message.

    This is the loop's decision policy for offline demos. It follows the
    escalation SOP: search first; if the KB refuses (no_answer), open a
    high-priority CRM ticket and post to #support-escalations; then wrap up."""

    _name = "demo-scripted-llm"

    def __init__(self) -> None:
        super().__init__(fn=self._decide)

    def _decide(self, messages) -> str:
        last = messages[-1]["content"]
        goal_question = next(
            (m["content"][len("GOAL:"):].strip()
             for m in messages
             if m["content"].startswith("GOAL:")),
            "customer question")
        if last.startswith("GOAL:"):
            question = last[len("GOAL:"):].strip()
            return json.dumps({
                "thought": "Search the knowledge base before answering.",
                "action": "search_kb",
                "args": {"query": question},
            })
        if last.startswith("OBSERVATION:"):
            payload = json.loads(last[len("OBSERVATION:"):].strip())
            if not payload.get("ok"):
                return json.dumps({
                    "thought": "The tool failed; stop and report.",
                    "action": "final_answer",
                    "answer": "I could not complete this request because a "
                              "tool failed.",
                })
            result = payload["result"]
            if isinstance(result, dict) and "answer" in result:
                if result.get("no_answer"):
                    return json.dumps({
                        "thought": "The KB has no answer; per the escalation "
                                   "SOP, open a high-priority CRM ticket.",
                        "action": "create_crm_ticket",
                        "args": {
                            "summary": "Question outside the knowledge base: "
                                       + goal_question[:120],
                            "priority": "high",
                            "confirm": True,
                        },
                    })
                answer = str(result["answer"])
                return json.dumps({
                    "thought": "The KB answered with citations.",
                    "action": "final_answer",
                    "answer": answer,
                })
            if isinstance(result, int):
                return json.dumps({
                    "thought": "Ticket created; escalate to the support "
                               "channel per the SOP.",
                    "action": "post_teams_message",
                    "args": {
                        "channel": "#support-escalations",
                        "text": f"Escalation: high-priority ticket {result} "
                                "needs an owner (question outside the KB).",
                        "confirm": True,
                    },
                })
            if isinstance(result, dict) and "channel" in result:
                return json.dumps({
                    "thought": "Escalation complete; summarise for the "
                               "customer.",
                    "action": "final_answer",
                    "answer": "I could not find that in our knowledge base, "
                              "so I opened a high-priority support ticket and "
                              "escalated it to the support team in "
                              "#support-escalations. Our team will follow up.",
                })
        return json.dumps({
            "thought": "Unexpected state; stop safely.",
            "action": "final_answer",
            "answer": "I could not process this request.",
        })


def run_scenario(loop, goal, label):
    print("=" * 72)
    print(f"SCENARIO {label}")
    print(f"GOAL: {goal}")
    print("=" * 72)
    result = loop.run(goal)
    for entry in result.trace:
        print(f"[step {entry['step']}] {entry['ts']} "
              f"action={entry.get('action', entry.get('error', 'parse'))}")
        if "thought" in entry:
            print(f"   thought: {entry['thought']}")
        if "args" in entry:
            print(f"   args: {json.dumps(entry['args'], ensure_ascii=False)}")
        if "observation" in entry:
            print(f"   observation: "
                  f"{json.dumps(entry['observation'], ensure_ascii=False)[:220]}")
        if "error" in entry:
            print(f"   error: {entry['error']}")
    print(f"\nSTEPS USED: {result.steps_used}  completed={result.ok}")
    print(f"FINAL ANSWER: {result.final_answer}\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true",
                        help="use the configured provider instead of the "
                             "scripted demo fake")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if not args.live:
        settings = settings.replace(llm_provider="fake")
    configure_logging(settings.log_level)
    settings.apply_seed()
    settings.logs_dir and Path(settings.logs_dir).mkdir(parents=True,
                                                         exist_ok=True)

    pipeline = load_or_build_pipeline(settings)
    crm = CrmMock(settings.crm_db_path)
    teams = TeamsMock(settings.teams_log_path)
    registry = build_default_registry(pipeline, crm, teams, user_role="guest")
    llm = get_provider(settings, fake=not args.live)
    if not args.live:
        # the pipeline's fake generator cites the chunk it ACTUALLY retrieved
        # (a real, valid citation id) instead of a placeholder id
        def _citing_answer(messages):
            content = messages[-1]["content"] if messages else ""
            match = re.search(r"\[source:([0-9a-f]{12})\]", content)
            if match:
                return ("Under the cancellation policy the refund amount "
                        "depends on how many days before the event start "
                        "date the customer cancels; the cited chunk holds "
                        f"the exact windows. [source:{match.group(1)}]")
            return "The knowledge base does not cover this question."

        pipeline.generator = FakeLLM(fn=_citing_answer)
        llm = DemoFakeLLM()
        print("LLM: scripted DemoFakeLLM (labelled mock; --live for a real "
              "provider)")
    else:
        print(f"LLM: live provider {llm.name}")
    loop = AgentLoop(registry, llm, max_steps=args.max_steps)

    before_tickets = crm.count()
    run_scenario(loop, SCENARIO_IN_SCOPE, "A - in scope (KB answer)")
    run_scenario(loop, SCENARIO_OUT_OF_KB, "B - out of KB (escalate)")

    print("=" * 72)
    print("AFTERMATH")
    print(f"CRM tickets created this run: {crm.count() - before_tickets}")
    for ticket in crm.list_tickets(limit=5):
        print(f"  - #{ticket['id']} [{ticket['priority']}] {ticket['status']}: "
              f"{ticket['summary'][:80]}")
    print(f"Teams messages posted: {len(teams.messages)} "
          f"(log: {teams.log_path})")
    for message in teams.messages:
        print(f"  - {message['channel']}: {message['text'][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
