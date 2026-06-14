# Adaptive AI Orchestration System: Overall Project Report

## 1. Project Purpose

The sole purpose of this project is to act as an intelligent gateway between a
user and multiple Large Language Models (LLMs).

Instead of sending every request to the largest and most expensive model, the
system examines each query and selects a model that should provide an acceptable
answer while balancing:

- response quality;
- response latency;
- inference cost;
- query difficulty;
- the amount of historical evidence available for each model.

The system also supports company-document questions through Retrieval-Augmented
Generation (RAG), evaluates generated answers after they are returned, and uses
the evaluation results to update future routing metrics.

In one sentence:

> The project is an adaptive multi-model LLM gateway designed to reduce cost and
> latency without unnecessarily sacrificing answer quality.

## 2. Problem Being Solved

Using one powerful LLM for every query is simple, but inefficient:

- Simple questions do not need an expensive large model.
- Complex questions may receive poor answers from a small model.
- Company-policy questions require private document context, not only general
  model knowledge.
- Model performance can change by query type, so fixed routing rules eventually
  become inaccurate.
- Without tracking, teams cannot prove whether routing saves money or preserves
  quality.

This project addresses those problems by combining query classification,
utility-based model selection, RAG, fallback execution, evaluation, adaptive
metric updates, persistence, security controls, and monitoring.

## 3. High-Level Architecture

```text
User
  |
  v
HTML Chat Interface
  |
  v
FastAPI API
  |
  +--> JWT authentication and rate limiting
  +--> Query classification preview
  +--> Security inspection and sanitization
  +--> Intent and complexity analysis
  +--> Weighted model selection
  +--> Optional RAG retrieval
  +--> Prompt construction
  +--> AWS Bedrock model execution with fallback
  +--> Query, cost, token, and audit persistence
  |
  +--> Immediate response to user
  |
  +--> In-process background evaluation
         |
         +--> LLM-as-judge quality score
         +--> Online running-average metric update

PostgreSQL/Supabase --> Streamlit Monitoring Dashboard
```

## 4. Complete Request Lifecycle

The main lifecycle is implemented in `Backend/main.py` inside the `/api/chat`
endpoint.

1. **Validate the request**
   - Pydantic verifies the query and session ID lengths.
   - JWT verification identifies the user.
   - SlowAPI limits requests by client IP.

2. **Preview routing**
   - The query is classified before security inspection so the security guard
     knows whether document-specific guardrails should apply.

3. **Inspect security**
   - The security guard checks query length, removes selected dangerous patterns,
     detects common prompt-injection phrases, and blocks selected out-of-scope
     RAG requests.
   - Blocked requests are written to the audit log.

4. **Classify the clean query**
   - Intent detection classifies it as general, company-specific, or unknown.
   - Complexity analysis classifies it as low, medium, or high.
   - The orchestrator chooses a `fast` or `reasoning` prompt strategy and marks
     whether RAG retrieval is needed.

5. **Select a model**
   - The decision engine reads historical metrics for each model at the detected
     complexity level.
   - It computes a complexity-aware weighted utility score.
   - It selects the highest-scoring candidate.

6. **Retrieve documents when needed**
   - The query is embedded with Amazon Titan Text Embeddings V2.
   - FAISS searches normalized document vectors.
   - Results below the similarity threshold are removed.
   - Relevant chunks are assembled into context.

7. **Build and execute the prompt**
   - A short prompt template is selected for fast, reasoning, or RAG execution.
   - The selected AWS Bedrock model is called.
   - If it fails, the execution layer tries an ordered fallback chain.

8. **Measure and persist**
   - End-to-end latency is measured.
   - Exact token metadata is used when available; otherwise token usage is
     estimated from text length.
   - Estimated cost is calculated using a local pricing table.
   - Query details and an audit record are committed to PostgreSQL.

9. **Return the response**
   - The answer is returned immediately with model, strategy, latency, and query
     ID metadata.
   - `quality_score` is initially `0.0` because evaluation has not finished.

10. **Evaluate and learn in the background**
    - FastAPI `BackgroundTasks` calls the LLM judge.
    - The evaluation is saved.
    - The selected model's quality and latency metrics are updated using an
      incremental running average.

## 5. Models and Their Roles

| Internal key | Actual Bedrock model | Intended role |
|---|---|---|
| `nova-micro` | Amazon Nova Micro | Fast and inexpensive answers |
| `llama3-8b` | Meta Llama 3.1 8B Instruct | Balanced reasoning |
| `haiku` | Meta Llama 3.3 70B Instruct | Complex answers and evaluation |

The `haiku` key is misleading because it does not map to Claude Haiku. It should
be renamed to something such as `llama3-70b`.

## 6. Core Decision Logic

### Intent Detection

`core/intent_detector.py` determines whether a query is general or related to
the organization's documents.

It uses a hybrid approach:

- deterministic phrases that are always general;
- deterministic phrases that are always company-specific;
- organization markers such as `our` or `this company`;
- HR topics such as leave, salary, policy, and resignation;
- sentence-transformer similarity as a tiebreaker.

Purpose: prevent general questions from unnecessarily activating RAG while still
recognizing company-policy questions.

### Complexity Analysis

`core/complexity_analyzer.py` assigns `low`, `medium`, or `high`.

It uses:

- query length;
- weighted reasoning keywords;
- high-complexity domain overrides;
- medium-complexity floor phrases;
- simple-question starter penalties;
- multi-part query bonuses.

It also activates RAG through a keyword list. The supplied `intent` currently
does not affect `_check_rag`, so RAG activation is primarily keyword-based.

### Execution Strategy

`core/orchestrator.py` combines intent and complexity results.

- Low complexity becomes `fast`.
- Medium and high complexity become `reasoning`.
- If retrieval is needed, the stored execution label becomes `fast+rag` or
  `reasoning+rag`.

The orchestrator does not select the model. This separation keeps query analysis
independent from data-driven model routing.

### Weighted Model Routing

`core/decision_engine.py` calculates:

```text
raw_score =
    quality_weight * p_quality
    - latency_weight * p_latency
    - cost_weight * p_cost

final_score = confidence(sample_count) * raw_score
```

Low-complexity queries prioritize cost and latency. High-complexity queries
prioritize quality.

Confidence grows logarithmically from `0.30` toward `1.00` as sample count
increases. Ties prefer the cheaper model.

Purpose: make routing interpretable and sensitive to both query difficulty and
historical model performance.

### Online Metric Update

`Learning/learning.py` updates quality and normalized latency:

```text
alpha = 1 / (1 + sample_count)
new_value = old_value + alpha * (observed_value - old_value)
```

Latency is normalized against a fixed 6,000 ms reference. Cost is not updated.

This is an online incremental running average. It is not a complete Bayesian
learning algorithm, reinforcement-learning policy, or contextual bandit.

## 7. Purpose of Every Project Area

### Root Files

| File or folder | Sole purpose |
|---|---|
| `Readme.md` | Provides setup, architecture, endpoint, and usage documentation. Some descriptions still call the router Bayesian and should be corrected. |
| `INTERVIEW_PREP_MASTER_PROMPT.md` | Provides a seven-day study and mock-interview plan for explaining and defending the project. |
| `PROJECT_OVERALL_REPORT.md` | Gives an implementation-grounded report of the entire project. |
| `requirements.txt` | Lists Python runtime dependencies. |
| `.env` | Stores local secrets and environment-specific configuration. It must never be committed or exposed. |
| `.gitignore` | Declares files Git should not track. |

### `Backend/`: Public API and Request Coordination

| File | Sole purpose |
|---|---|
| `Backend/main.py` | Creates the FastAPI app and coordinates the complete request lifecycle, endpoints, token/cost calculation, persistence, audit logging, and background evaluation. |
| `Backend/schemas.py` | Defines Pydantic request and response contracts so invalid API data is rejected early. |
| `Backend/dependencies.py` | Defines optional API-key verification and the IP-based SlowAPI rate limiter. The current endpoints use JWT, and `verify_api_key` is imported but not applied. |
| `Backend/__init__.py` | Marks the directory as a Python package. |

API endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Confirms API and database availability. |
| `POST /api/auth/register` | Creates an in-memory user account. |
| `POST /api/auth/login` | Verifies credentials and returns a JWT. |
| `GET /api/auth/me` | Returns the authenticated user's details. |
| `POST /api/chat` | Runs the full orchestration and generation pipeline. |
| `POST /api/feedback` | Stores a one-to-five rating and optional comment. |
| `GET /api/metrics` | Returns a small aggregate system summary. |
| `GET /api/probabilities` | Returns current routing metrics. |

### `core/`: Query Understanding and Routing

| File | Sole purpose |
|---|---|
| `core/intent_detector.py` | Classifies general versus company-specific intent using rules and semantic similarity. |
| `core/complexity_analyzer.py` | Estimates difficulty and determines whether RAG should be activated. |
| `core/orchestrator.py` | Combines intent, complexity, retrieval need, and prompt strategy into one routing description. |
| `core/decision_engine.py` | Scores available models and selects the candidate with the highest confidence-adjusted utility. |
| `core/__init__.py` | Marks the directory as a Python package. |

### `RAG/`: Company Document Retrieval

| File or folder | Sole purpose |
|---|---|
| `RAG/document_loader.py` | Extracts text from PDFs and splits it into 500-character chunks with 50-character overlap. |
| `RAG/embedder.py` | Generates Titan embeddings for document chunks and user queries. |
| `RAG/vector_store.py` | Builds, saves, loads, and searches a FAISS `IndexFlatIP` vector index. |
| `RAG/retriever.py` | Filters search results at a `0.75` similarity threshold and builds source-labelled context. |
| `RAG/__init__.py` | Marks the directory as a Python package. |
| `documents/` | Stores the source company-policy and handbook PDFs used by RAG. |
| `faiss_index/index.faiss` | Stores the searchable FAISS vector index. |
| `faiss_index/metadata.pkl` | Maps vector positions to chunk text and source names. It must only be loaded from a trusted source because pickle can execute malicious code. |

FAISS uses inner-product search after L2 normalization. For normalized vectors,
inner product corresponds to cosine similarity.

### `Execution/`: Prompting and Model Calls

| File | Sole purpose |
|---|---|
| `Execution/prompt_builder.py` | Builds compact fast, reasoning, and context-grounded RAG prompts while limiting context size. |
| `Execution/Bedrock_client.py` | Converts common requests into provider-specific AWS Bedrock payloads and extracts text and token metadata. It also calls Titan embeddings. |
| `Execution/Execution_layer.py` | Executes the selected model, applies output-token budgets, and attempts fallback models when calls fail. |
| `Execution/__init__.py` | Marks the directory as a Python package. |

Fallback order:

| Primary | Fallback order |
|---|---|
| `nova-micro` | `llama3-8b`, then `haiku` |
| `llama3-8b` | `nova-micro`, then `haiku` |
| `haiku` | `llama3-8b`, then `nova-micro` |

### `Evaluation/`: Quality Measurement

| File | Sole purpose |
|---|---|
| `Evaluation/Evaluator.py` | Uses Llama 3.3 70B as an LLM judge, extracts structured JSON, detects possible hallucinations, and computes a weighted quality score. |
| `Evaluation/golden_tests.py` | Stores known handbook question-answer examples and checks whether evaluation behavior remains acceptable. |

Quality formula:

```text
quality =
    0.40 * correctness
    + 0.35 * relevance
    + 0.25 * completeness
    - hallucination_penalty
```

The hallucination penalty is `0.05` per flag and is capped at `0.20`.

### `Learning/`: Adaptive Updates

| File | Sole purpose |
|---|---|
| `Learning/learning.py` | Updates per-model, per-complexity quality and latency metrics after successful evaluations. |
| `Learning/__init__.py` | Marks the directory as a Python package. |

### `database/`: Persistence

| File | Sole purpose |
|---|---|
| `database/connection.py` | Builds the PostgreSQL SQLAlchemy engine, session factory, ORM base, and FastAPI session dependency. |
| `database/models.py` | Defines the relational schema and ORM relationships. |
| `database/crud.py` | Centralizes database creation, lookup, update, feedback, audit, and aggregation operations. |
| `database/init_db.py` | Creates all declared tables. |
| `database/seed.py` | Inserts initial routing metrics for every model and complexity pair. |
| `database/migrate_add_tokens.py` | Adds token and estimated-cost columns to an existing `queries` table. |

Database entities:

| Table | Purpose |
|---|---|
| `queries` | Stores each query, answer, routing decision, latency, fallback status, tokens, and cost. |
| `evaluations` | Stores one LLM-judge result per query. |
| `probabilities` | Stores quality, latency, cost, and sample count for each model-complexity pair. |
| `audit_logs` | Stores request and blocked-request security/audit events. |
| `feedback` | Stores one user rating and comment per query. |
| `users` | Defines a persistent user schema, although the active API currently uses the in-memory user store instead. |

### `auth/`: Authentication

| File | Sole purpose |
|---|---|
| `auth/auth_handler.py` | Hashes passwords with bcrypt and creates, decodes, and verifies 24-hour HS256 JWTs. |
| `auth/user_store.py` | Provides the active in-memory user database and seeds a demo user at process startup. |
| `auth/__init__.py` | Marks the directory as a Python package. |

The SQLAlchemy user CRUD functions and `users` table exist, but registration and
login currently use `auth/user_store.py`. Therefore, registered users disappear
when the API process restarts and are not shared across multiple server workers.

### `Security/`: Input Guardrails

| File | Sole purpose |
|---|---|
| `Security/Security_Guard.py` | Checks input length, removes selected dangerous patterns, blocks common direct prompt-injection phrases, and applies limited RAG scope restrictions. |
| `Security/__init__.py` | Marks the directory as a Python package. |

These checks are defense-in-depth. Regex cannot reliably stop paraphrased,
encoded, multilingual, indirect, or previously unknown prompt-injection attacks.

### `tracking/`: Analytics Queries

| File | Sole purpose |
|---|---|
| `tracking/metrics_tracker.py` | Aggregates model usage, quality, latency, strategy, complexity, token, cost, and routing metrics for the dashboard. |
| `tracking/__init__.py` | Marks the directory as a Python package. |

### `Dashboard/`: Operational Monitoring

| File | Sole purpose |
|---|---|
| `Dashboard/app.py` | Creates a Streamlit dashboard that reads PostgreSQL metrics and displays routing, performance, quality, token, and cost charts. |
| `Dashboard/__init__.py` | Marks the directory as a Python package. |

The dashboard directly accesses the database rather than calling the FastAPI
metrics endpoints.

### `Frontend/`: User Interface

| File | Sole purpose |
|---|---|
| `Frontend/index.html` | Provides a standalone login/register/chat interface, stores the JWT in browser local storage, calls the API, and displays routing metadata. |
| `Frontend/logo.png` | Stores a visual logo asset. |

The frontend includes a plaintext demo API key, but the backend chat endpoint
does not currently enforce API-key verification.

### `tests/`: Automated Checks

| File | Sole purpose |
|---|---|
| `tests/test_suite.py` | Runs lightweight tests for security rules, complexity, intent, orchestration, evaluator helpers, authentication, and schema validation. |
| `tests/__init__.py` | Marks the directory as a Python package. |

Most current tests are deterministic helper tests. They do not fully test the
database, FastAPI endpoints, Bedrock integration, concurrency, fallback behavior,
or complete RAG retrieval pipeline.

## 8. Security Design

Implemented controls:

- bcrypt password hashing;
- signed JWT access tokens with issued-at and expiration claims;
- request validation with Pydantic;
- IP-based rate limiting;
- restricted CORS origin list;
- prompt-injection regex checks;
- selected input sanitization;
- RAG scope blocking;
- audit logging.

Important limitations:

- The JWT secret has an insecure fallback value if the environment variable is
  missing.
- The active user store is in memory.
- API-key verification exists but is not attached to endpoints.
- SHA-256 alone is appropriate only for high-entropy API keys, not passwords.
- The rate limiter is local to one process unless configured with distributed
  storage.
- Sanitizing SQL-like text is not a substitute for parameterized queries.
- Retrieved documents are not inspected for indirect prompt injection.
- JWTs have no refresh, revocation, role, or permission model.

## 9. Reliability and Observability

Implemented reliability features:

- model fallback chains;
- database connection pre-ping;
- response latency tracking;
- token and estimated-cost tracking;
- audit records;
- LLM quality evaluation;
- dashboard summaries;
- evaluation failure fallback behavior.

Missing production reliability features:

- explicit model-call timeouts;
- retry policy with exponential backoff and jitter;
- circuit breakers;
- provider-level failover;
- durable background workers;
- idempotent evaluation jobs;
- distributed locks or atomic metric updates;
- centralized structured logging;
- tracing and request correlation IDs;
- alerting and service-level objectives;
- caching and load shedding.

## 10. Important Implementation Limitations

1. **The learning update is not Bayesian.**  
   It is an incremental mean-like running average.

2. **The router is greedy.**  
   It always chooses the highest current score and has no systematic exploration.

3. **Confidence scaling can suppress all metrics during cold start.**  
   Seed values exist, but every row begins with the same sample-count confidence.

4. **Metric normalization needs calibration.**  
   Quality, latency, and cost values are placed on roughly comparable scales
   through assumptions, not through a validated normalization study.

5. **RAG activation is heuristic.**  
   Keyword matching can miss relevant questions and incorrectly trigger on
   unrelated ones.

6. **RAG retrieval is basic.**  
   It uses exact flat-vector search and threshold filtering but has no reranker,
   query rewriting, hybrid lexical search, answer citations, or retrieval
   evaluation integrated into routing.

7. **Background tasks are not durable.**  
   If the API process stops after responding, evaluation and learning can be
   lost.

8. **Probability updates can race.**  
   Concurrent updates can read the same sample count and overwrite each other.

9. **User persistence is disconnected.**  
   The database defines users, but the running API uses an in-memory store.

10. **The API response omits useful routing details.**  
    The frontend expects complexity, selected model, and fallback metadata, but
    `ChatResponse` currently returns only response, strategy, used model, latency,
    initial quality, and query ID.

11. **All-model failure is stored like an answer.**  
    The execution layer returns an error message instead of raising a service
    error, so it can be persisted and evaluated as though it were a valid answer.

12. **Documentation contains stale claims.**  
    `Readme.md` describes Bayesian/product-based routing, while the current code
    uses weighted additive utility scoring and a running average.

## 11. Recommended Production Evolution

### Highest Priority

- Persist users in PostgreSQL and remove the insecure JWT fallback secret.
- Rename `haiku` to accurately identify Llama 3.3 70B.
- Move evaluation to a durable queue such as SQS, Celery, or Kafka.
- Add timeouts, retries, circuit breakers, and clear failure status handling.
- Make metric updates atomic and concurrency-safe.
- Calibrate metric normalization and validate routing against a fixed benchmark.

### Routing Improvements

- Add exploration using epsilon-greedy, upper-confidence-bound routing, or a
  contextual bandit.
- Include query features, RAG status, token estimate, and user feedback as
  routing context.
- Compare adaptive routing against fixed-model baselines.
- Track quality-adjusted cost and latency as formal business metrics.

### RAG Improvements

- Add hybrid semantic and keyword retrieval.
- Add a reranker.
- Preserve page numbers and produce answer citations.
- Detect indirect prompt injection in retrieved content.
- Evaluate retrieval recall, precision, faithfulness, and answer correctness
  separately.
- Replace pickle metadata with a safer serialization format.

### Platform Improvements

- Use asynchronous Bedrock calls or worker pools for better throughput.
- Add Redis-backed distributed rate limiting and caching.
- Add structured logs, tracing, dashboards, and alerts.
- Add database indexes for common dashboard and lookup queries.
- Use Alembic for schema migrations.
- Add unit, integration, contract, security, load, and end-to-end tests.

## 12. How to Explain the Project in an Interview

### Short Version

> I built an adaptive multi-model LLM gateway that routes each query to an AWS
> Bedrock model based on query complexity and historical quality, latency, and
> cost metrics. It activates a FAISS-based RAG pipeline for company-policy
> questions, uses model fallbacks for reliability, tracks tokens and cost in
> PostgreSQL, and evaluates answers in the background to update future routing
> metrics. The current implementation is an interpretable heuristic router with
> online running-average updates, not a full Bayesian or contextual-bandit
> system.

### Main Value

The project demonstrates how an AI platform can treat model selection as an
engineering optimization problem instead of assuming one model is best for every
request.

### Central Trade-Off

The system gains transparency and low implementation complexity by using rules
and weighted utility scoring, but it sacrifices the adaptability, exploration,
and statistical guarantees that a calibrated contextual-bandit system could
provide.

## 13. Final Assessment

This project is a strong end-to-end prototype of an LLM orchestration platform.
Its most important achievement is not merely calling multiple models; it connects
classification, model selection, RAG, fallback execution, persistence,
evaluation, learning, security, and monitoring into one understandable system.

Its current design is best described as:

> An interpretable, heuristic, adaptive LLM-routing prototype with RAG and
> operational tracking.

It should not yet be described as a production-grade autonomous router or a fully
Bayesian learning system. The most valuable next step is to validate the routing
policy experimentally, make evaluation durable, and introduce controlled
exploration with stronger production reliability.
