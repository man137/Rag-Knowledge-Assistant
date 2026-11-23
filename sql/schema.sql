-- schema.sql — production-shaped schema for the query metrics.
-- Mirrors observability/metrics_store.py (which auto-creates the same table
-- so the repo works out of the box); this file is the reviewable definition.

CREATE TABLE IF NOT EXISTS query_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT    NOT NULL,          -- ISO 8601 UTC timestamp
    request_id         TEXT    NOT NULL,          -- correlation id (also the
                                                   -- feedback key)
    question_hash      TEXT    NOT NULL,          -- sha1(question)[:16]; the
                                                   -- raw question is NEVER
                                                   -- stored
    mode               TEXT    NOT NULL,          -- dense | bm25 | hybrid |
                                                   -- hybrid_rerank
    prompt_version     TEXT    NOT NULL,          -- v1 | v2 | v3
    user_role          TEXT    NOT NULL,          -- guest | staff | manager |
                                                   -- hr
    latency_ms         REAL    NOT NULL DEFAULT 0,
    ttft_ms            REAL    NOT NULL DEFAULT 0, -- time to first token
    tokens_in          INTEGER NOT NULL DEFAULT 0,
    tokens_out         INTEGER NOT NULL DEFAULT 0,
    no_answer          INTEGER NOT NULL DEFAULT 0, -- 1 = refused below
                                                   -- confidence threshold
    invalid_citations  INTEGER NOT NULL DEFAULT 0, -- count of cited chunk
                                                   -- ids absent from context
    feedback           TEXT,                       -- 'up' | 'down' | NULL
    CHECK (no_answer IN (0, 1)),
    CHECK (feedback IN ('up', 'down') OR feedback IS NULL)
);

CREATE INDEX IF NOT EXISTS idx_query_log_mode       ON query_log(mode);
CREATE INDEX IF NOT EXISTS idx_query_log_ts         ON query_log(ts);
CREATE INDEX IF NOT EXISTS idx_query_log_request_id ON query_log(request_id);
