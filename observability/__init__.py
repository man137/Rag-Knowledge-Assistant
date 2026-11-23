"""observability — JSON logging and SQLite metrics for queries.

WHAT: package marker for logging_setup.py (JSON-formatted stdlib logging with
request-id propagation via contextvars and PII redaction) and metrics_store.py
(the SQLite query_log that powers /metrics and the analytics SQL).
WHY: the JD lists "logging and monitoring" and "evaluation of accuracy /
latency / response quality" — you cannot evaluate latency you never recorded.
MATHS: none.
FAILURE MODE: logs are redacted by REGEX (see rag.security honest limits);
anything the patterns miss still reaches the log stream, so never log raw
customer data in the first place.
INTERVIEW ONE-LINER: "Every query is one JSON log line and one metrics row
with a request-id that follows it end to end — that is what makes p50/p95
per mode possible later."
"""

__version__ = "1.0.0"
