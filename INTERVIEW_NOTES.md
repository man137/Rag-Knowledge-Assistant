# INTERVIEW_NOTES — how to defend every line of this project

Read order: architecture walkthrough -> design decisions -> failure modes ->
the 40 questions. Everything here maps to a file you own, so quote code when
answering. If you cannot explain a line, that file is the study plan.

---

## 1. Architecture walkthrough (2 minutes, say it like this)

"An ingest path and a serving path. Ingestion loads six synthetic markdown
policies with front-matter roles, cleans them, chunks them with a
sentence-aware packer, embeds them with a small pre-trained model, and saves
a NumPy matrix plus a BM25 inverted index. Serving is one pipeline object:
the question is length-checked and scanned for injection phrases, the role
is turned into an allow-set, then hybrid retrieval fuses dense cosine and
BM25 ranks with Reciprocal Rank Fusion, optionally reranked by a
cross-encoder. A no-answer check on the top score can skip the LLM entirely
and return a refusal. Surviving chunks are packed into a versioned prompt
under a token budget — lowest-ranked chunks dropped first — the model
streams an answer, and every `[source:chunk_id]` citation is validated
against what was actually in the prompt. Per-stage timings, tokens and flags
go to a SQLite metrics store with the question hashed, never raw. On top of
that sit a FastAPI service with SSE streaming and a hand-written
plan-act-observe agent whose write tools need an explicit confirm=true."

## 2. Twelve design decisions and their trade-offs

1. **Chunk size (500 tokens default; swept 256/512/768)** — bigger chunks
   carry more context but dilute the embedding and waste prompt budget;
   smaller chunks retrieve sharper but can cut an answer in half. The sweep
   in `run_eval.py` measures recall/MRR instead of arguing.
2. **Overlap (80 tokens)** — prevents an answer from straddling a boundary;
   costs duplicated storage and some redundancy in the prompt. Fixed
   chunker has EXACT overlap; the sentence packer overlaps whole sentences
   up to the budget.
3. **Cosine via dot product** — vectors are L2-normalised at encode time, so
   ranking is a single matrix-vector product; division is done once at
   encoding, not per query per chunk. The store still normalises the query
   defensively.
4. **BM25 vs dense** — BM25 nails exact tokens ("INR 9,000", "FORM-12") and
   is fully explainable; dense handles paraphrase. Neither wins everywhere —
   out-of-scope questions with exact terms fool BM25, paraphrases fool the
   lexical half.
5. **Why hybrid** — the two error modes are different, so fusion covers both.
   Fusion is also where managed services landed (Azure AI Search hybrid),
   which validates the design.
6. **RRF (k=60)** — rank-based fusion needs no score calibration across
   incomparable scales (cosine in [0,1] vs unbounded BM25). Worked example
   in the test: a chunk 1st+3rd scores 1/61+1/63=0.03227, one 2nd+2nd
   scores 2/62=0.03226 — ranks alone decide.
7. **Reranking as a separate stage behind a flag** — a cross-encoder is
   more accurate but costs one forward pass per (query, chunk) pair, so it
   only sees the shortlist (~20 -> ~4). Flag off by default: honest about
   cost on a laptop.
8. **MMR** — relevance minus redundancy: pick argmax of
   `λ·sim(q,d) − (1−λ)·max_{s∈S} sim(d,s)`; trades a little relevance for
   diversity when a document repeats itself. Component kept out of the
   default path at this corpus size (say so — it shows judgment).
9. **No-answer threshold (mode-aware)** — dense compares cosine (max 1.0);
   BM25 compares absolute score (scale depends on corpus); hybrid uses the
   DENSE score of the top fused chunk because RRF always has a rank 1 and
   cannot say "nothing is relevant"; rerank uses the cross-encoder score.
   Below threshold the LLM is skipped and the canonical refusal is returned.
10. **Citation validation** — answers cite `[source:chunk_id]`; the
    generator validates each id against the context actually sent
    (after budget drops), so a hallucinated citation is reported, not
    trusted. That is what makes RAG auditable.
11. **ACL before ranking** — forbidden rows get −inf BEFORE
    `np.argpartition`, so a restricted chunk cannot occupy a top-k slot at
    all; filtering afterwards would leak ranking information and risk
    implementation mistakes. Eval access-control items assert zero leakage.
12. **Prompt versioning (v1/v2/v3)** — prompts are files with a contract
    (grounded citations in v2, injection defence + answer format in v3);
    `prompt_ab.py` measures each on the same set. Versioning makes prompts
    testable code instead of folklore.

## 3. Failure modes (know the symptom AND the mitigation)

- **Wrong chunk retrieved** — right document, wrong section; the answer is
  confidently off. Mitigations: hybrid retrieval, rerank, citations that
  let the reader check, groundedness metric exposing it.
- **Lost in the middle** — LLMs attend less to mid-context text. Mitigation:
  few chunks (k=4), best-first packing, budget drops from the tail.
- **Stale index** — document changes, index doesn't; answers cite a
  `chunk_id` that no longer exists after re-ingest. Mitigation: explicit
  ingest step + rebuild; chunk ids are stable per (source, start_char) so
  drift is visible.
- **Prompt injection** — corpus or user text carries instructions. Layers:
  phrase heuristics that flag/strip (documented as incomplete), v3 prompt
  treats context as data, canary in the eval set proves the check works.
- **ACL leakage** — the worst enterprise failure: an HR answer to a guest.
  Layers: mask before ranking, flat role model (no accidental inheritance),
  eval items that must refuse, leak-rate metric asserted 0.
- **Tool-call hallucination** — the agent invents a tool or arguments.
  Layers: allowlist, schema validation, confirm=true for writes, one JSON
  repair attempt, step limit, errors fed back as observations.

## 4. Forty likely questions, with concise answers

### RAG and evaluation (20)

1. **What is RAG in one sentence?** Retrieve relevant passages from a
   trusted corpus and generate the answer grounded in them, so the model
   answers from your data with citations instead of parametric memory.
2. **Why not fine-tune?** Cheaper to update, auditable per answer,
   citations possible; fine-tuning changes style and knowledge but not
   facts-at-runtime and needs retraining per change.
3. **Walk through your ingest pipeline.** load (front-matter roles) ->
   clean (NFKC, strip repeated headers) -> chunk (sentence-aware, overlap,
   stable sha1 ids) -> embed (L2-normalised) -> store NumPy matrix + BM25
   index.
4. **Why chunk at all?** Embedding models compress; whole policies dilute
   the signal. Chunks trade completeness for specificity; overlap protects
   boundary answers.
5. **How did you pick 500 tokens?** I didn't guess — the eval sweeps
   256/512/768 across four retrieval modes and writes a table; on this
   corpus [MEASURED: see eval_out/results.csv after your real run].
6. **Explain BM25's formula.** idf(t) = ln(1+(N−df+0.5)/(df+0.5)); term
   score = idf · tf·(k1+1)/(tf + k1·(1−b+b·dl/avgdl)). k1 saturates
   repetition, b scales length normalisation; mine are 1.5/0.75.
7. **What is RRF and why k=60?** score = Σ 1/(k+rank); rank-only fusion of
   incomparable score scales. k=60 from the Cormack et al. paper; it
   dampens the difference between ranks 1 and 2 without flattening the tail.
8. **Dense vs BM25 vs hybrid — when do you use which?** Exact identifiers
   → bm25; paraphrase → dense; production → hybrid plus a reranker, with
   the threshold calibrated per mode.
9. **Why L2-normalise vectors?** Dot product then equals cosine, so
   ranking is one matrix multiply; normalisation cost is paid once per
   document instead of per query.
10. **How does IVF work?** Seeded k-means clusters the vectors; queries
    compare to nprobe nearest centroids and score only those inverted
    lists — sub-linear at the cost of recall, measured in
    scripts/benchmark.py.
11. **What is MRR vs nDCG?** MRR cares about the first correct answer;
    nDCG discounts later positions with log2 and normalises by the ideal
    ordering — better when several gold chunks exist.
12. **What is groundedness in your eval?** Fraction of answer sentences
    whose max cosine to a retrieved chunk ≥ 0.6 — a PROXY for support, not
    truth; it rewards copying and punishes correct-but-unquoted answers, so
    it is reported alongside refusal accuracy, leak rate and citation
    validity.
13. **How do you make the model refuse out-of-scope questions?** A
    retrieval-confidence threshold that skips the LLM and emits a canonical
    refusal, plus a prompt rule that answers only from context. Measured by
    refusal accuracy on 8 out-of-scope items.
14. **What good is a refusal the model can't verify?** The refusal is
    driven by retrieval confidence (measurable, thresholded), not by the
    model's feeling; that's why the check runs BEFORE generation.
15. **How do you know citations are real?** Each cited chunk_id is checked
    against the ids actually packed into the prompt; invalid ones are
    reported in the response and counted in metrics (invalid_citation_rate).
16. **Your eval set — how do you trust it?** 50 items in four categories
    (30 in-scope, 8 out-of-scope, 6 injection, 6 access-control); a
    validator in CI checks schema, exact counts and that every gold keyword
    literally exists in the gold source.
17. **What did prompt v2 change and how do you know it helped?** v2 adds
    citations and the refusal contract; v3 adds injection defence and
    format rules; prompt_ab.py measures all three on the same items —
    [MEASURED: run `make prompt-ab` without --fake and read the table].
18. **How is latency measured?** Per-stage timings plus time-to-first-token
    on every ask, percentiles in metrics.py and six SQL queries for p50/p95
    per mode.
19. **What breaks when the corpus grows 100x?** Brute-force dot product
    first (switch to IVF or AI Search), then the prompt budget (more chunks
    means more drops), then the fake-based tests' calibration (irrelevant —
    tests only pin behaviour, not quality).
20. **What would you do differently?** Stemming/lemmatisation for BM25,
    per-user ACL instead of four flat roles, and a human-labeled eval set
    instead of keyword proxies.

### Azure and enterprise integration (10)

21. **How would you do this with Azure AI Search?** Same architecture,
    managed pieces: vector index for my matrix, BM25+vector hybrid with the
    service's fusion, semantic ranker for my cross-encoder, an OData
    search.in() filter for my ACL mask — my `azure_ai_search_retriever.py`
    implements exactly that behind the same Retriever interface.
22. **And with Azure OpenAI?** Generator and embedder swap through
    `azure_adapters/`; the deployment-name-vs-model-name difference is
    handled and documented in the adapter docstring.
23. **Where does Copilot Studio fit?** It calls my service: openapi.json ->
    custom connector (API-key auth on X-API-Key) -> topic action hitting
    /ask_sync, because connectors cannot consume my SSE stream. Full steps
    in COPILOT_STUDIO_GUIDE.md.
24. **Power Automate vs your agent loop?** Declarative low-code orchestration
    vs my hand-written plan-act-observe loop. Flows are great for fixed
    sequences; my loop handles model-chosen branching with safety rails
    (allowlist, confirm, step limit).
25. **How would you authenticate this properly?** Entra ID bearer tokens;
    the API validates against the tenant and derives the ROLE from token
    claims — the X-User-Role header is explicitly a demo stand-in
    (see api/auth.py's honest docstring).
26. **Where do secrets live?** Key Vault + managed identity in production;
    locally .env (gitignored) with dummy values in .env.example; the Docker
    image bakes none.
27. **Application Insights?** My JSON logs already carry ts/level/logger/
    msg/request_id with PII redaction; they map one-to-one onto
    App Insights telemetry, and the SQLite query_log becomes a managed
    store with the same six analytics queries.
28. **What's your CRM/Teams story?** The agent opens a ticket and posts an
    escalation through tools with an allowlist and confirm=true; mocks
    implement the interfaces a Dynamics connector and Teams webhook would
    fill (`agent/crm_mock.py`, `azure_adapters/teams_webhook.py`).
29. **What changes in production?** Swap the retriever class, real auth,
    Key Vault, managed logging, scheduled re-ingest from SharePoint via
    Graph, re-run the eval before trusting any number — the honest list is
    in azure_adapters/README_AZURE.md.
30. **Have you deployed any of this to Azure?** No. Adapters are interfaces
    tested against mocks and labelled NOT TESTED AGAINST LIVE SERVICE; I
    can describe the deployment (Container Apps) without claiming I did it.

### Agents and tool calling (5)

31. **Explain your loop.** Prompt (system protocol + tool schemas + goal) ->
    model emits one JSON action -> registry checks allowlist, schema, and
    confirm -> execute -> observation back -> repeat until final_answer or
    the step limit; every step is timestamped in a trace.
32. **What if the model emits garbage JSON?** Exactly one repair attempt:
    the error is fed back as an observation and the retry is its own step;
    a second failure ends the run with ok=false.
33. **How do you stop destructive actions?** Write tools require
    confirm=true in the arguments, the allowlist is the registry itself,
    unknown arguments are rejected, and everything else comes back as an
    error observation the model must recover from.
34. **What is a tool-call hallucination and where is yours handled?** The
    model invents a tool or args; caught at the registry (allowlist +
    schema validation) and at the loop (repair, step limit), visible in the
    trace for debugging.
35. **Why not function-calling APIs?** They are model-specific; my JSON
    protocol works on any provider (the same FakeLLM tests the mechanics),
    and writing it taught me what those APIs actually do.

### Behavioural (5) — STAR with YOUR blanks

36. **A time you debugged something hard.** S: [project/coursework].
    T: [what was broken, e.g. "tests failed only in CI"]. A: [how you
    isolated it, e.g. "bisected the failure to a fixture, reproduced
    minimally, fixed the root cause not the symptom"]. R: [outcome +
    what you now do differently].
37. **A decision you made with incomplete information.** S: []. T: [].
    A: [e.g. "chose thresholds by measurement on a small set, documented
    them as provisional"]. R: [].
38. **Working with a stakeholder/teammate who disagreed.** S: []. T: [].
    A: []. R: [].
39. **Learning something quickly for a deadline.** S: []. T: []. A: []
    (this project's honest answer works: "I learned enough Azure service
    semantics to design the adapters, and labelled what I didn't test").
    R: [].
40. **A failure and what you learned.** S: []. T: []. A: []. R: [].
    (A real one from building this repo counts — e.g. a threshold that
    refused everything until you measured the score distribution.)

## 5. Ten questions to ask the interviewer

1. Which processes does the team automate first with Copilot Studio —
   event operations, client services, or internal knowledge?
2. How much of the current RAG stack is managed (AI Search) vs custom code?
3. How do you evaluate agent answers today — is there an eval set, or is it
   manual review?
4. Where does the knowledge base live today — SharePoint, CRM records, or
   documents the team maintains?
5. Who owns access control for agent users — Entra ID groups, CRM roles?
6. What does the path from this internship to a full-time role look like?
7. What does a typical support escalation look like when a copilot
   misbehaves in front of a client?
8. How is the team structured between prompt/agent work and integration
   work with Power Platform?
9. What was the last thing a copilot project got badly wrong, and what
   changed because of it?
10. What would a successful intern have shipped by the end?

## 6. Claims I must NOT make

- That any accuracy/latency number is "X%" before running the real eval on
  my own hardware and filling the [MEASURED] fields.
- That I deployed anything to Azure, or tested a live Azure service.
- That the mocks are integrations, or that the Teams/CRM/Graph adapters saw
  production traffic.
- That the security heuristics make the system "secure" (they are a first
  layer; the docstring says so).
- That the synthetic corpus means anything statistically about real corpora.
- That this repo uses Copilot Studio or Power Platform beyond writing the
  guide (no tenant was used).
- That the agent in live mode is production-grade — it is a demo of
  mechanics bounded by safety rails.
- Any employer data, salary, headcount or "production" wording on the resume.
