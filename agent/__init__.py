"""agent — hand-written enterprise agent layer (no agent framework).

WHAT: package marker for the tool registry (tools.py), the SQLite CRM and
in-memory Teams mocks (crm_mock.py, teams_mock.py) and the plan-act-observe
loop (loop.py).
WHY: the JD asks for "AI agent workflows" and "enterprise tool integration";
building the loop by hand (JSON tool-calls, allowlist, confirm gating,
observation feedback) demonstrates the mechanics that Copilot Studio and the
function-calling APIs hide behind a button.
MATHS: none.
FAILURE MODE: the loop trusts only the allowlist; every other failure (bad
JSON, bad args, tool error) becomes an observation the model can recover
from, and a step limit bounds runaway loops.
INTERVIEW ONE-LINER: "I hand-wrote the agent loop — schema-validated JSON
tool calls, an allowlist, human-confirm gating on writes, one repair attempt
— so I can explain exactly what a framework like Copilot Studio generates."
"""

__version__ = "1.0.0"
