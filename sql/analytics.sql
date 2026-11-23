-- analytics.sql — 6 operational queries over query_log.
-- Note on percentiles: plain SQLite has no PERCENTILE_CONT. These queries use
-- the NEAREST-RANK method with window functions (SQLite >= 3.25):
-- p50 = the value at rank ceil(0.50 * n), p95 = the value at rank
-- ceil(0.95 * n). The Python summary in observability/metrics_store.py uses
-- linear interpolation instead, so the two agree closely but not exactly.

-- (1) p50/p95 latency by retrieval mode -------------------------------
WITH ranked AS (
    SELECT mode, latency_ms,
           ROW_NUMBER() OVER (PARTITION BY mode ORDER BY latency_ms) AS rn,
           COUNT(*)    OVER (PARTITION BY mode)                      AS n
    FROM query_log
)
SELECT mode,
       COUNT(*) AS queries,
       MAX(CASE WHEN rn = CAST(CEIL(0.50 * n) AS INTEGER) THEN latency_ms END) AS p50_ms,
       MAX(CASE WHEN rn = CAST(CEIL(0.95 * n) AS INTEGER) THEN latency_ms END) AS p95_ms
FROM ranked
GROUP BY mode
ORDER BY mode;

-- (2) no-answer (refusal) rate by day -----------------------------------
SELECT date(ts) AS day,
       COUNT(*)                                        AS queries,
       ROUND(AVG(no_answer), 4)                        AS no_answer_rate
FROM query_log
GROUP BY date(ts)
ORDER BY day;

-- (3) top unanswered question hashes (repeated refusal patterns) --------
SELECT question_hash,
       COUNT(*) AS times_unanswered,
       MIN(date(ts)) AS first_seen,
       MAX(date(ts)) AS last_seen
FROM query_log
WHERE no_answer = 1
GROUP BY question_hash
ORDER BY times_unanswered DESC, question_hash
LIMIT 10;

-- (4) thumbs feedback ratio by prompt version ---------------------------
SELECT prompt_version,
       COUNT(*)                                                        AS answered_with_feedback,
       SUM(CASE WHEN feedback = 'up'   THEN 1 ELSE 0 END)              AS ups,
       SUM(CASE WHEN feedback = 'down' THEN 1 ELSE 0 END)              AS downs,
       ROUND(1.0 * SUM(CASE WHEN feedback = 'up' THEN 1 ELSE 0 END)
                   / NULLIF(SUM(CASE WHEN feedback IS NOT NULL
                                    THEN 1 ELSE 0 END), 0), 4)         AS up_ratio
FROM query_log
WHERE feedback IS NOT NULL
GROUP BY prompt_version
ORDER BY prompt_version;

-- (5) invalid-citation rate (audited citation hallucinations) -----------
SELECT COUNT(*) AS queries_with_citations,
       SUM(CASE WHEN invalid_citations > 0 THEN 1 ELSE 0 END) AS queries_with_invalid,
       ROUND(1.0 * SUM(CASE WHEN invalid_citations > 0 THEN 1 ELSE 0 END)
                   / NULLIF(COUNT(*), 0), 4) AS invalid_citation_rate
FROM query_log
WHERE tokens_out > 0;   -- only rows where the LLM actually answered

-- (6) daily volume (capacity planning) ----------------------------------
SELECT date(ts) AS day,
       COUNT(*) AS queries,
       ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
       SUM(tokens_in + tokens_out) AS tokens_total
FROM query_log
GROUP BY date(ts)
ORDER BY day;
