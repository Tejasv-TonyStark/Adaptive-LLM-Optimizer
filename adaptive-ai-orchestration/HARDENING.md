# Routing and reliability changes

These changes address the implementation weaknesses found during review. They do
not establish real-world accuracy, savings, or immunity to prompt injection.
Existing uncommitted project changes were retained. No database migration is
needed: reliability uses the existing usage ledger and audit details use JSON.

## Behavior changes

- Unknown task wording now produces medium complexity with `uncertain=true`,
  rather than automatically selecting the cheapest route. Elementary integer
  parity demonstrations and constrained scheduling have regression coverage.
  Difficulty remains a heuristic; unseen wording can still be misclassified.
- Production model eligibility is low: all models; medium: 8B or 70B; high: 70B.
  These are conservative configured assumptions, not measured capability claims.
  Failover stays within this set. Exhausting suitable models returns HTTP 503.
- Adaptive selection subtracts a recent failure penalty and performs 5% bounded
  exploration by default. Only suitable, healthy alternatives under the estimated
  exploration call-cost ceiling qualify. High complexity has one eligible model,
  so does not explore. Static remains the deployment default.
- Health uses the latest 20 generation attempts from the last 30 minutes. Three
  consecutive infrastructure failures open a 60-second circuit. Validation failures
  affect the failure rate but do not trip the infrastructure circuit. After cooldown,
  calls can resume. This is not a distributed single-probe half-open implementation;
  simultaneous requests may retry together. Estimated cost is not a hard billing cap.
- Learning starts as a seeded online mean and transitions to an exponentially
  weighted average with minimum alpha 0.1. Old observations no longer make new
  observations negligible. This is decay by observation count, not wall-clock age.
- Judge scores remain visible, but learning is disabled unless a current independent
  calibration report allows that judge/answer-model pair. A model never trains from
  judging its own answer. Missing, invalid or stale reports fail closed. Human user
  ratings remain feedback, not automatically trusted training labels.
- Follow-ups use at most three completed turns belonging to the authenticated user
  and session. Question/answer fields are bounded to 1,000 characters each. The
  topic anchor is retained across longer follow-up chains. General generation gets
  the bounded history; retrieval uses the topic plus follow-up. Prior answers never
  become document evidence. Ambiguous references without history request clarification.
  Reference detection is rule-based and can still miss complex discourse or topic shifts.
- Credential-request screening runs regardless of routing, and benign role requests
  such as “Act as a Python tutor” are accepted. Input is checked again with retrieval
  intent available. Regex screening is not the document authorization mechanism.
- RAG uses an extractive JSON evidence contract. The server checks citation IDs
  against actual retrieved chunks and verifies that each quote occurs verbatim in
  that chunk. Only verified quotes are rendered. Fabricated quotes, unknown IDs,
  malformed evidence and token-truncated output trigger suitable-model fallback;
  invalid evidence that cannot be recovered produces abstention.
- Exact quotation does **not** prove that a passage answers the question or includes
  every exception. Extractive output trades synthesis for traceability. General
  nonempty answers are not guaranteed correct; there is no automatic truth oracle
  or uncalibrated per-request judge escalation.

## Configuration

See `.env.example`. Leave `ROUTING_POLICY=static` until comparative evaluation is
available; set it to `adaptive` to enable health-adjusted utility and exploration.
Set `EXPLORATION_RATE=0` to disable exploration. The default
`MAX_EXPLORATION_COST_USD=0.002` bounds estimated input/output generation cost for
an exploration choice, not the whole request, embeddings, retries or judges.

`RAG_ACCESS_POLICY` may name a deployment-owned JSON file:

```json
{"handbook.pdf": ["*"], "private-policy.pdf": ["1", "7"]}
```

Keys are exact source filenames and values are string user IDs; `*` means all
authenticated users. Unlisted sources and users are denied. Invalid/unreadable
policy files fail closed. Filtering happens before selecting top-k. With no policy
configured the corpus is shared by all authenticated users, as in the original app.
Do not place differently restricted documents in the shared corpus without an ACL.
The flat index is searched fully when an ACL is active; a larger corpus should use
an index/database with native filtered search.

## Verification and evaluation

Windows commands (use `python` inside an activated environment):

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_regressions tests.test_hardening -q
.\venv\Scripts\python.exe -m tests.test_suite
.\venv\Scripts\python.exe -m Evaluation.benchmark --output benchmark_results/classification.json
.\venv\Scripts\python.exe -m Evaluation.benchmark --dataset Evaluation/routing_cases.json --output benchmark_results/routing_regressions.json
```

The original 32 cases are a small smoke set, not an independent accuracy estimate.
The new 12 routing cases were used during development and are explicitly labeled
development data. Keep future independently authored cases separate from tuning.
Tests mock provider calls; SQLite tests do not establish PostgreSQL concurrency or
live Bedrock behavior.

The following commands incur provider charges and were not run during local
verification:

```powershell
.\venv\Scripts\python.exe -m Evaluation.benchmark --live --output benchmark_results/live.json
.\venv\Scripts\python.exe -m Evaluation.retrieval_benchmark --live --threshold 0.75 --top-k 5 --user-id 1
```

The live comparison now includes fixed cheapest, middle and strongest models plus
static and adaptive policies on the same cases. Each case has a separate session
so previous benchmark questions cannot leak into follow-up context. Forced fixed
model experiments intentionally bypass production capability floors. The report
includes actual learned-update counts: an adaptive run without calibrated judge
feedback still uses seed quality assumptions. Repeat independently labeled trials
before interpreting performance differences.

The retrieval evaluator measures supporting-passage recall and empty retrieval on
unanswerable questions separately. It can replay saved observations without paid
calls using `--observations observations.json`; input maps each case ID to
`{"context": "...", "chunks": [], "found": true}`. A high similarity score is
not proof of answerability. Use development cases to tune retrieval, then evaluate
untouched cases. The provided source snippets come from the existing dataset and
must be checked against the installed document versions.

## Enabling calibrated learning

Collect independent human ratings on real answers before enabling learning. Do
not derive the human ratings from the judge or show the judge's ratings to labelers.
Prepare a JSON list with this shape; replace the example with real reviewed data:

```json
[
  {"id":"review-001", "judge_model":"llama3-70b", "answer_model":"nova-micro",
   "human_quality":0.8, "judge_quality":0.75, "human_reviewed":true}
]
```

Run `python -m Evaluation.calibration labels.json --output benchmark_results/calibration.json`
and set `JUDGE_CALIBRATION_PATH` to that report. Eligibility requires at least 20
unique reviewed examples **per judge/answer-model pair**, mean absolute error at
most 0.15, absolute mean bias at most 0.10, matching rubric version and report age
under 30 days. These operational thresholds are initial policy choices, not a
statistical guarantee. With the default 70B judge, 70B answers cannot train the
router. Changing `JUDGE_MODEL` requires calibration for the new pairs. Labels and
reports are trusted operator inputs; never allow clients to upload an eligibility
report as authorization to train.

## Remaining empirical work

Live provider compatibility, workload-specific capability floors, judge/human
agreement, retrieval relevance, costs and latency under load need real measurements.
No human labels were fabricated, no deployed configuration was changed, and no
production database was migrated. The safeguards and evaluation tooling are ready;
the project should not claim that these empirical limitations are already solved.
