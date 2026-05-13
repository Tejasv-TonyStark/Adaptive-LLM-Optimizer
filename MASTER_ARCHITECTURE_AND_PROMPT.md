# ADAPTIVE AI ORCHESTRATION & OPTIMIZATION SYSTEM
## Complete Architecture Document + Claude Opus Master Prompt

---

# PART 1 — COMPLETE SYSTEM ARCHITECTURE

---

## 1.1 SYSTEM OVERVIEW

```
ONE unified chatbot interface.
THREE internal execution strategies.
THREE dedicated models.
ZERO manual model selection by user.
FULL adaptive probabilistic routing.
FULL evaluation and learning loop.
```

---

## 1.2 HIGH LEVEL ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                   (Streamlit Chat UI)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │ POST /api/chat
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Auth Layer  │  │ Rate Limiter │  │ Input Validator   │  │
│  │ (API Key)   │  │ (SlowAPI)    │  │ (Pydantic)        │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION ENGINE                      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              INTENT DETECTOR                          │   │
│  │   general / specific / unknown                        │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │              COMPLEXITY ANALYZER                      │   │
│  │   low / medium / high + retrieval_needed flag         │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │              DECISION ENGINE                          │   │
│  │   Reads probability tables from PostgreSQL            │   │
│  │   Calculates: 0.6*quality - 0.3*latency - 0.1*cost   │   │
│  │   Selects: best model + strategy                      │   │
│  └──────────────────────┬───────────────────────────────┘   │
└───────────────────────────────────────────────────────────┬──┘
                            │                               │
              ┌─────────────┼──────────────┐               │
              ▼             ▼              ▼               │
┌─────────────────┐ ┌──────────────┐ ┌───────────────┐    │
│  STRATEGY 1     │ │  STRATEGY 2  │ │  STRATEGY 3   │    │
│  Fast Mode      │ │  Reasoning   │ │  RAG Mode     │    │
│  TinyLlama      │ │  Phi3 Mini   │ │  Mistral 7B   │    │
│                 │ │              │ │  + FAISS      │    │
│  ~300-600ms     │ │  ~1s-2.5s    │ │  ~4s-7s       │    │
└────────┬────────┘ └──────┬───────┘ └──────┬────────┘    │
         └────────────────┬┘────────────────┘             │
                          │                               │
                          ▼                               │
┌─────────────────────────────────────────────────────────┐   │
│                  RESPONSE RETURNED TO USER               │◄──┘
└─────────────────────────┬───────────────────────────────┘
                          │ (background async)
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  EVALUATION ENGINE                       │
│   Judge: Phi3 Mini                                       │
│   Metrics: relevance, correctness, completeness          │
│   For RAG: passes document chunks to judge also          │
│   Output: quality_score (0.0 - 1.0)                     │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  LEARNING ENGINE                         │
│   Formula: P_new = P_old + alpha*(actual - P_old)        │
│   Alpha: 1/(1+sample_count) — adaptive learning rate     │
│   Updates: p_quality, p_latency in PostgreSQL            │
│   Cost: NEVER updates (hardware fixed)                   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  POSTGRESQL DATABASE                     │
│   Tables: queries, evaluations, probabilities,           │
│           audit_logs, feedback                           │
│   Host: Supabase Free Tier                               │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  STREAMLIT DASHBOARD                     │
│   Strategy usage, model usage, latency trends,           │
│   quality trends, probability tables live,               │
│   routing explanations                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 1.3 MODULE BREAKDOWN

---

### MODULE 1 — FASTAPI BACKEND

**File:** `backend/main.py`

**Endpoints:**
```
POST /api/chat          → main query endpoint
GET  /api/metrics       → fetch system metrics
POST /api/feedback      → user feedback on responses
GET  /api/health        → system health check
GET  /api/probabilities → current probability table
```

**Responsibilities:**
- Receive and validate requests
- API key authentication via header
- Rate limiting via SlowAPI (10 req/min per key)
- Input sanitization via Pydantic
- Write audit log on every request
- Call orchestration engine
- Return response to user

**Request Schema:**
```python
class ChatRequest(BaseModel):
    query: str          # min 3 chars, max 1000 chars
    session_id: str     # for conversation tracking

class ChatResponse(BaseModel):
    response: str
    strategy_used: str
    model_used: str
    latency_ms: int
    quality_score: float  # added async after evaluation
```

---

### MODULE 2 — INTENT DETECTOR

**File:** `core/intent_detector.py`

**Purpose:**
Detect whether user wants general knowledge or company-specific information BEFORE checking keywords.

**Logic:**
```python
SPECIFIC_KEYWORDS = [
    "our", "my", "this company", "here",
    "as per", "according to", "what does it say",
    "organization", "firm"
]

GENERAL_KEYWORDS = [
    "generally", "typically", "usually",
    "in general", "define", "what is meant by",
    "explain what", "tell me about"
]

# Semantic intent also checked via embedding similarity
# to two anchor sentences:
GENERAL_ANCHOR = "What does X mean in general terms?"
SPECIFIC_ANCHOR = "What does our company policy say about X?"
```

**Output:**
```python
{
    "intent": "general" / "specific" / "unknown",
    "confidence": 0.85
}
```

---

### MODULE 3 — COMPLEXITY ANALYZER

**File:** `core/complexity_analyzer.py`

**Purpose:**
Estimate query complexity using rule-based scoring.

**Scoring System:**
```python
score = 0

# Factor 1: Query length
if word_count < 8:   score += 1   # Low signal
if word_count 8-15:  score += 2   # Medium signal
if word_count > 15:  score += 3   # High signal

# Factor 2: Reasoning keywords
REASONING_KEYWORDS = [
    "compare", "contrast", "explain", "analyze",
    "difference", "vs", "versus", "why", "how does",
    "evaluate", "discuss", "elaborate"
]
score += count_of_reasoning_keywords * 2

# Factor 3: Question structure
if starts_with("what is", "define", "who is"):
    complexity_bias = "low"
if starts_with("compare", "explain", "analyze"):
    complexity_bias = "high"

# Final classification
if score <= 3:   complexity = "low"
if score 4-7:    complexity = "medium"
if score >= 8:   complexity = "high"
```

**RAG Trigger Check (2 layers):**
```python
# Layer 1: Keyword match
RAG_KEYWORDS = [
    "policy", "policies", "clause", "section",
    "guideline", "reimbursement", "leave", "salary",
    "benefits", "allowance", "resignation", "notice period",
    "security protocol", "handbook", "compliance",
    "procedure", "confidential", "data protection"
]

# Layer 2: Semantic similarity to FAISS index
similarity_score = faiss_search(query_embedding)
retrieval_needed = (keyword_match AND similarity > 0.70)
                   OR (similarity > 0.85)  # very high similarity = RAG regardless
```

**Combined with Intent:**
```python
if intent == "general":
    retrieval_needed = False  # override always

if intent == "specific":
    retrieval_needed = True if similarity > 0.60  # lower threshold
```

**Output:**
```python
{
    "complexity": "low" / "medium" / "high",
    "retrieval_needed": True / False,
    "intent": "general" / "specific" / "unknown",
    "confidence": 0.87,
    "reasoning": "reasoning keywords detected: compare, vs"
}
```

---

### MODULE 4 — DECISION ENGINE

**File:** `core/decision_engine.py`

**Purpose:**
Select optimal model and strategy using probability scoring.

**Probability Scoring Formula:**
```
score = (0.6 * p_quality) - (0.3 * p_latency) - (0.1 * p_cost)
```

**Initial Probability Table (Cold Start Priors):**

From MMLU benchmarks + hardware profiling + model size knowledge:

```
TinyLlama (MMLU=25.9%, RAM=2GB, avg_latency=400ms)
─────────────────────────────────────────────────
complexity=low:    p_quality=0.82, p_latency=0.08, p_cost=0.13
complexity=medium: p_quality=0.46, p_latency=0.11, p_cost=0.13
complexity=high:   p_quality=0.26, p_latency=0.15, p_cost=0.13

Phi3 Mini (MMLU=68.8%, RAM=4GB, avg_latency=1500ms)
─────────────────────────────────────────────────
complexity=low:    p_quality=0.90, p_latency=0.25, p_cost=0.27
complexity=medium: p_quality=0.88, p_latency=0.38, p_cost=0.27
complexity=high:   p_quality=0.69, p_latency=0.63, p_cost=0.27

Mistral 7B (MMLU=62.5%, RAM=8GB, avg_latency=4000ms)
─────────────────────────────────────────────────
complexity=low:    p_quality=0.90, p_latency=0.63, p_cost=0.53
complexity=medium: p_quality=0.83, p_latency=0.88, p_cost=0.53
complexity=high:   p_quality=0.63, p_latency=1.00, p_cost=0.53
```

**Decision Logic:**
```python
def select_model(complexity, retrieval_needed, intent):

    # RAG override — always Mistral for company docs
    if retrieval_needed and intent != "general":
        return {"strategy": "RAG", "model": "mistral"}

    # Score all models
    scores = {}
    for model in ["tinyllama", "phi3", "mistral"]:
        probs = db.get_probabilities(model, complexity)
        scores[model] = (
            0.6 * probs.p_quality
          - 0.3 * probs.p_latency
          - 0.1 * probs.p_cost
        )

    best_model = max(scores, key=scores.get)

    strategy_map = {
        "tinyllama": "Fast",
        "phi3": "Reasoning",
        "mistral": "Reasoning"  # if mistral wins on non-RAG
    }

    return {
        "strategy": strategy_map[best_model],
        "model": best_model,
        "scores": scores,
        "reason": f"Highest score: {scores[best_model]:.4f}"
    }
```

---

### MODULE 5 — EXECUTION LAYER

**File:** `core/execution_layer.py`

**Purpose:**
Execute selected strategy. Load model on demand. Handle fallback.

**Model Loading (On Demand):**
```python
# Never load all 3 models simultaneously
# i7 15GB RAM cannot handle all 3 loaded
# Load → Execute → Response (model stays loaded for 5min idle)

FALLBACK_CHAIN = {
    "mistral":   "phi3",
    "phi3":      "tinyllama",
    "tinyllama": None
}

def execute_with_fallback(model, prompt):
    try:
        return ollama.chat(model=model, messages=[...])
    except Exception as e:
        fallback = FALLBACK_CHAIN[model]
        if fallback:
            return ollama.chat(model=fallback, messages=[...])
        return {"error": "System temporarily unavailable"}
```

**Strategy Prompts:**

Fast Mode (TinyLlama):
```
Answer this clearly and briefly:
{query}
```

Reasoning Mode (Phi3 Mini):
```
You are an expert assistant. Think step by step.

Question: {query}

Step 1 - Understand what is being asked
Step 2 - Break down key concepts
Step 3 - Provide a structured, complete answer
```

RAG Mode (Mistral):
```
You are a company policy assistant.

STRICT RULES:
1. Answer ONLY using the context provided below.
2. If the answer is NOT in the context, respond:
   "This information is not available in the company knowledge base."
3. Do NOT use your own general knowledge.
4. Do NOT reveal full document contents.
5. Do NOT answer non-policy questions.

Context:
{retrieved_chunks}

Question: {query}

Answer strictly based on context:
```

---

### MODULE 6 — RAG SYSTEM

**File:** `rag/rag_engine.py`

**Components:**
- `rag/document_loader.py` — load and chunk PDF/text documents
- `rag/embedder.py` — generate embeddings via sentence-transformers
- `rag/vector_store.py` — FAISS index management
- `rag/retriever.py` — similarity search and chunk retrieval

**Pipeline:**
```
Company Documents (PDF/TXT)
         ↓
Chunking (500 tokens, 50 overlap)
         ↓
Embedding (sentence-transformers: all-MiniLM-L6-v2)
         ↓
FAISS Index stored locally
         ↓
─────────────── QUERY TIME ───────────────
Query → Embed → FAISS Search → Top 3 Chunks
         ↓
Chunks injected into Mistral prompt
         ↓
Grounded response generated
```

**Relevance Check:**
```python
SIMILARITY_THRESHOLD = 0.70

distances, indices = faiss_index.search(query_vec, k=3)
top_score = distances[0][0]

if top_score < SIMILARITY_THRESHOLD:
    return "NOT_FOUND"  # triggers guardrail response
```

**Document Security:**
```
All documents stored locally.
FAISS index stored locally.
Never sent to external APIs.
Only top 3 relevant chunks passed to model.
Full documents never exposed.
```

---

### MODULE 7 — EVALUATION ENGINE

**File:** `evaluation/evaluator.py`

**Judge Model:** Phi3 Mini
**Runs:** Asynchronously after user receives response

**Judge Prompt:**
```
You are an expert AI response evaluator.

Question: {query}
Response: {response}
{f"Source Document Context: {chunks}" if is_rag else ""}

Evaluate strictly. Score 0.0 to 1.0 for each.

Return ONLY valid JSON:
{{
    "relevance": 0.0,
    "correctness": 0.0,
    "completeness": 0.0,
    "reasoning": "brief explanation"
}}
```

**Quality Score Formula:**
```python
quality_score = (
    0.40 * correctness   # most important — wrong = useless
  + 0.35 * relevance     # must answer the question
  + 0.25 * completeness  # partial answer still has value
)
```

**Self-Evaluation Bias Handling:**
```
Phi3 evaluating Phi3 responses → known limitation
Documented in system notes
Mitigated by: async evaluation, not blocking decisions
Future: use TinyLlama for Phi3 evaluation (cross-evaluation)
```

---

### MODULE 8 — LEARNING ENGINE

**File:** `learning/learning_engine.py`

**Formula:**
```
P_new = P_old + alpha * (actual - P_old)

alpha = 1 / (1 + sample_count)   ← adaptive rate
```

**What Updates:**
```
p_quality → from evaluation engine score
p_latency → from actual measured latency (normalized)
p_cost    → NEVER updates (hardware fixed)
```

**Normalization of Latency:**
```python
LATENCY_CEILING = 4000  # ms
actual_normalized = min(1.0, actual_ms / LATENCY_CEILING)
```

**Alpha Behavior:**
```
sample_count=0   → alpha=1.00  (fully trust new data)
sample_count=9   → alpha=0.10  (balanced)
sample_count=49  → alpha=0.02  (mostly trust history)
sample_count=99  → alpha=0.01  (very stable)
```

---

### MODULE 9 — METRICS TRACKER

**File:** `tracking/metrics_tracker.py`

**Tracks per query:**
```python
{
    "query_text":     str,
    "complexity":     "low/medium/high",
    "intent":         "general/specific/unknown",
    "strategy":       "Fast/Reasoning/RAG",
    "model_used":     "tinyllama/phi3/mistral",
    "latency_ms":     int,
    "quality_score":  float,
    "relevance":      float,
    "correctness":    float,
    "completeness":   float,
    "eval_reasoning": str,
    "timestamp":      datetime,
    "session_id":     str,
    "fallback_used":  bool
}
```

---

### MODULE 10 — POSTGRESQL DATABASE

**Host:** Supabase Free Tier
**ORM:** SQLAlchemy

**Table 1: queries**
```sql
CREATE TABLE queries (
    id            SERIAL PRIMARY KEY,
    session_id    TEXT,
    query_text    TEXT NOT NULL,
    complexity    TEXT NOT NULL,
    intent        TEXT,
    strategy      TEXT NOT NULL,
    model_used    TEXT NOT NULL,
    response      TEXT NOT NULL,
    latency_ms    INTEGER,
    fallback_used BOOLEAN DEFAULT FALSE,
    timestamp     TIMESTAMPTZ DEFAULT NOW()
);
```

**Table 2: evaluations**
```sql
CREATE TABLE evaluations (
    id            SERIAL PRIMARY KEY,
    query_id      INTEGER REFERENCES queries(id),
    relevance     REAL,
    correctness   REAL,
    completeness  REAL,
    quality_score REAL,
    reasoning     TEXT,
    timestamp     TIMESTAMPTZ DEFAULT NOW()
);
```

**Table 3: probabilities**
```sql
CREATE TABLE probabilities (
    id            SERIAL PRIMARY KEY,
    model         TEXT NOT NULL,
    complexity    TEXT NOT NULL,
    p_quality     REAL DEFAULT 0.5,
    p_latency     REAL DEFAULT 0.5,
    p_cost        REAL,
    sample_count  INTEGER DEFAULT 0,
    last_updated  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model, complexity)
);
```

**Table 4: audit_logs**
```sql
CREATE TABLE audit_logs (
    id          SERIAL PRIMARY KEY,
    event_type  TEXT NOT NULL,
    detail      JSONB,
    api_key     TEXT,
    ip_address  TEXT,
    timestamp   TIMESTAMPTZ DEFAULT NOW()
);
```

**Table 5: feedback**
```sql
CREATE TABLE feedback (
    id          SERIAL PRIMARY KEY,
    query_id    INTEGER REFERENCES queries(id),
    rating      INTEGER CHECK (rating BETWEEN 1 AND 5),
    comment     TEXT,
    timestamp   TIMESTAMPTZ DEFAULT NOW()
);
```

---

### MODULE 11 — DASHBOARD

**File:** `dashboard/app.py`

**Panels:**
```
1. System Overview
   - Total queries today
   - Strategy breakdown (Fast/Reasoning/RAG) %
   - Average quality score today

2. Performance Metrics
   - Avg latency per strategy (line chart)
   - Quality score trends over time (line chart)
   - Model usage pie chart

3. Live Probability Table
   - Current p_quality per model per complexity
   - Current p_latency per model per complexity
   - Sample counts

4. Routing Explainability
   - Last 10 queries
   - What model was selected and why
   - Score breakdown shown

5. Evaluation Insights
   - Relevance / Correctness / Completeness averages
   - Judge reasoning log
```

---

### MODULE 12 — SECURITY LAYER

**File:** `security/auth.py`, `security/guardrails.py`

```
API Key Auth    → X-API-Key header required
Rate Limiting   → SlowAPI: 10 requests/minute per key
Input Validation → Pydantic: min 3 chars, max 1000 chars
Audit Logging   → every request logged to PostgreSQL
Guardrails      → RAG refuses out-of-scope questions
Content Filter  → basic prompt injection detection
```

---

## 1.4 COMPLETE FOLDER STRUCTURE

```
adaptive-ai-orchestration/
│
├── backend/
│   ├── main.py                    # FastAPI app, all endpoints
│   ├── dependencies.py            # Auth, rate limit dependencies
│   └── schemas.py                 # Pydantic request/response models
│
├── core/
│   ├── orchestrator.py            # Main orchestration coordinator
│   ├── intent_detector.py         # General vs specific intent
│   ├── complexity_analyzer.py     # Low/medium/high classifier
│   └── decision_engine.py         # Probability-based model selector
│
├── execution/
│   ├── execution_layer.py         # Strategy executor + fallback
│   ├── prompt_builder.py          # Builds prompts per strategy
│   └── model_manager.py           # Ollama model loading/unloading
│
├── rag/
│   ├── rag_engine.py              # Main RAG coordinator
│   ├── document_loader.py         # Load + chunk documents
│   ├── embedder.py                # sentence-transformers wrapper
│   ├── vector_store.py            # FAISS index management
│   └── retriever.py               # Similarity search + chunk fetch
│
├── evaluation/
│   ├── evaluator.py               # LLM-as-Judge evaluation
│   └── scoring.py                 # Quality score calculation
│
├── learning/
│   └── learning_engine.py         # Bayesian probability updater
│
├── tracking/
│   └── metrics_tracker.py         # Metrics collection + storage
│
├── database/
│   ├── connection.py              # SQLAlchemy setup + Supabase
│   ├── models.py                  # ORM table definitions
│   ├── crud.py                    # All DB read/write operations
│   └── seed.py                    # Cold start probability seeding
│
├── security/
│   ├── auth.py                    # API key validation
│   └── guardrails.py              # RAG scope enforcement
│
├── dashboard/
│   └── app.py                     # Streamlit dashboard
│
├── documents/
│   └── policies/                  # Company policy PDFs/TXTs
│       ├── hr_policy.pdf
│       ├── reimbursement_policy.pdf
│       └── security_policy.pdf
│
├── faiss_index/
│   ├── index.faiss                # FAISS vector index
│   └── metadata.json             # Chunk metadata
│
├── config/
│   └── settings.py                # All config: thresholds, weights
│
├── tests/
│   ├── test_complexity_analyzer.py
│   ├── test_decision_engine.py
│   ├── test_rag_engine.py
│   ├── test_evaluation.py
│   └── test_learning_engine.py
│
├── .env                           # API keys, DB URL, secrets
├── requirements.txt
└── README.md
```

---

## 1.5 TECH STACK FINAL

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11+ |
| Backend | FastAPI + Uvicorn | Latest |
| Models Runtime | Ollama | Latest |
| Strategy 1 | TinyLlama | 1.1B |
| Strategy 2 | Phi-3 Mini | 3.8B |
| Strategy 3 | Mistral | 7B Q4 |
| Judge | Phi-3 Mini | 3.8B |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 |
| Vector Store | FAISS | Latest |
| Database | PostgreSQL | 15+ |
| DB Host | Supabase | Free Tier |
| ORM | SQLAlchemy | 2.0+ |
| Rate Limiting | SlowAPI | Latest |
| Validation | Pydantic | v2 |
| Dashboard | Streamlit | Latest |
| Version Control | Git + GitHub | — |
| Cost | 100% Free | — |

---

---

# PART 2 — CLAUDE OPUS MASTER PROMPT

---

Copy everything below this line and paste directly to Claude Opus:

---

```
You are an expert AI Systems Architect, Backend Engineer, 
and Software Engineering mentor.

I am building a portfolio-grade AI systems engineering project.
You must fully understand the architecture before helping me.

Read everything below completely before responding.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROJECT TITLE:
Adaptive AI Orchestration & Optimization System

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORE CONCEPT:

ONE chatbot interface to the user.
THREE internal execution strategies with THREE different models.
ZERO manual model selection — backend decides everything.

The system answers the question:
"What is the most optimal way to handle this query?"

based on:
- query complexity (low / medium / high)
- user intent (general knowledge vs company-specific)
- retrieval requirement (does it need company documents?)
- historical model performance (Bayesian probability tables)
- latency and resource cost

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THREE STRATEGIES AND THREE MODELS:

Strategy 1 — Fast Execution
Model: TinyLlama (1.1B)
For: Simple, low-complexity queries
Example: "What is Python?" / "Define DBMS"
Latency: ~300-600ms

Strategy 2 — Reasoning Mode
Model: Phi-3 Mini (3.8B)
For: Deep technical, reasoning-heavy queries
Example: "Compare Transformers vs RNN" / "Explain CAP theorem"
Latency: ~1s-2.5s

Strategy 3 — RAG Mode
Model: Mistral 7B Q4
For: Company policy document queries ONLY
Example: "What is reimbursement limit?" / "Explain leave policy"
Pipeline: Query → Embed → FAISS Search → Retrieve Chunks → Mistral
Latency: ~4s-7s
STRICT: Only answers from company documents. Refuses all other questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW ROUTING DECISIONS ARE MADE:

Step 1 — Intent Detection
Detects whether query is general knowledge or company-specific.
Uses keyword matching + semantic similarity to anchor sentences.
Output: general / specific / unknown

Step 2 — Complexity Analysis
Scores query using:
- word count
- reasoning keywords (compare, explain, analyze, vs)
- retrieval keywords (policy, reimbursement, leave, clause)
Output: low / medium / high + retrieval_needed flag

Step 3 — RAG Check (2 layers)
Layer 1: keyword match against RAG keyword list
Layer 2: semantic similarity of query against FAISS index
If intent=general → skip RAG regardless of keywords
If similarity > 0.70 AND keyword match → use RAG
Output: retrieval_needed = True / False

Step 4 — Decision Engine (Probabilistic)
If retrieval_needed = True → always route to RAG (Mistral)
If retrieval_needed = False → score all models:

score = (0.6 * p_quality) - (0.3 * p_latency) - (0.1 * p_cost)

Select model with highest score.
Probability tables stored in PostgreSQL.
Updated after every query via Bayesian update.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBABILITY SYSTEM (CRITICAL TO UNDERSTAND):

Every model has 3 probabilities per complexity level:

p_quality  = P(quality=high | model, complexity)
p_latency  = P(latency=high | model, complexity)
p_cost     = P(cost=high    | model)  ← NEVER updates

Initial values seeded from:
- p_quality: derived from MMLU benchmark scores
- p_latency: derived from hardware benchmarks on i7 15GB RAM
- p_cost:    derived from RAM usage / total RAM (static forever)

INITIAL PROBABILITY TABLE:

TinyLlama (MMLU=25.9%, RAM=2GB, avg=400ms):
  low:    p_quality=0.82, p_latency=0.08, p_cost=0.13
  medium: p_quality=0.46, p_latency=0.11, p_cost=0.13
  high:   p_quality=0.26, p_latency=0.15, p_cost=0.13

Phi3 Mini (MMLU=68.8%, RAM=4GB, avg=1500ms):
  low:    p_quality=0.90, p_latency=0.25, p_cost=0.27
  medium: p_quality=0.88, p_latency=0.38, p_cost=0.27
  high:   p_quality=0.69, p_latency=0.63, p_cost=0.27

Mistral 7B (MMLU=62.5%, RAM=8GB, avg=4000ms):
  low:    p_quality=0.90, p_latency=0.63, p_cost=0.53
  medium: p_quality=0.83, p_latency=0.88, p_cost=0.53
  high:   p_quality=0.63, p_latency=1.00, p_cost=0.53

UPDATE FORMULA (after every query):
alpha = 1 / (1 + sample_count)   ← adaptive learning rate
P_new = P_old + alpha * (actual - P_old)

p_quality updates from: evaluation engine quality score
p_latency updates from: actual measured latency (normalized: ms/4000)
p_cost NEVER updates.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVALUATION ENGINE (LLM AS JUDGE):

After response is generated:
1. Send query + response + (chunks if RAG) to Phi3 Mini as judge
2. Judge returns: relevance, correctness, completeness (0.0-1.0 each)
3. quality_score = 0.40*correctness + 0.35*relevance + 0.25*completeness
4. Score saved to PostgreSQL
5. Learning engine uses score to update probability tables

IMPORTANT: Evaluation runs ASYNCHRONOUSLY.
User receives response immediately.
Evaluation happens in background.
This prevents latency added to user experience.

For RAG evaluation:
Pass retrieved document chunks to judge also.
Judge uses source document as ground truth for correctness.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KNOWN ARCHITECTURE DECISIONS AND FIXES:

1. Models loaded ON DEMAND (never all 3 simultaneously)
   Reason: i7 15GB RAM cannot hold all 3 (14GB total)

2. Fallback chain implemented:
   mistral → phi3 → tinyllama → error message

3. RAG is STRICTLY scoped to company documents only
   Prompt guardrail + similarity threshold enforcement

4. Evaluation is async (background task)
   User never waits for judge to finish

5. p_cost is static (hardware never changes)

6. FAISS threshold = 0.70 (configurable in settings.py)

7. Intent detection prevents false RAG triggers
   "What are company policies generally?" → NOT RAG → Phi3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECH STACK:

Language:        Python 3.11+
Backend:         FastAPI + Uvicorn
Models:          Ollama (TinyLlama, Phi3:mini, Mistral:7b-q4)
Embeddings:      sentence-transformers (all-MiniLM-L6-v2)
Vector Store:    FAISS
Database:        PostgreSQL on Supabase Free Tier
ORM:             SQLAlchemy 2.0
Rate Limiting:   SlowAPI
Validation:      Pydantic v2
Dashboard:       Streamlit
Version Control: Git + GitHub
Cost:            100% FREE

Hardware Target: Intel i7, 15GB RAM, No GPU assumptions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOLDER STRUCTURE:

adaptive-ai-orchestration/
├── backend/         main.py, dependencies.py, schemas.py
├── core/            orchestrator.py, intent_detector.py,
│                    complexity_analyzer.py, decision_engine.py
├── execution/       execution_layer.py, prompt_builder.py,
│                    model_manager.py
├── rag/             rag_engine.py, document_loader.py,
│                    embedder.py, vector_store.py, retriever.py
├── evaluation/      evaluator.py, scoring.py
├── learning/        learning_engine.py
├── tracking/        metrics_tracker.py
├── database/        connection.py, models.py, crud.py, seed.py
├── security/        auth.py, guardrails.py
├── dashboard/       app.py
├── documents/       company policy PDFs
├── faiss_index/     FAISS index files
├── config/          settings.py
└── tests/           unit tests per module

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

POSTGRESQL TABLES:

queries        → query_text, complexity, intent, strategy,
                 model_used, response, latency_ms, fallback_used
evaluations    → query_id, relevance, correctness, completeness,
                 quality_score, reasoning
probabilities  → model, complexity, p_quality, p_latency,
                 p_cost, sample_count, last_updated
audit_logs     → event_type, detail (JSONB), api_key, timestamp
feedback       → query_id, rating (1-5), comment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECURITY:

API key authentication (X-API-Key header)
Rate limiting: 10 requests/minute per key (SlowAPI)
Input validation: Pydantic (min 3, max 1000 chars)
Audit logging: every request logged
RAG guardrail: refuses out-of-scope questions
Basic prompt injection detection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEVELOPMENT CONSTRAINTS:

- 100% free tooling only
- Local models only (Ollama)
- No GPU assumptions
- i7 laptop, 15GB RAM
- Must be prototype-friendly
- Must be hiring/portfolio quality
- 4 week timeline, 3 hours/day, 4 days/week

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT I NEED FROM YOU:

Help me implement this project module by module.

Follow this order:
1. Database setup (PostgreSQL + SQLAlchemy + seed probabilities)
2. FastAPI backend (endpoints + auth + rate limiting)
3. Complexity Analyzer + Intent Detector
4. Decision Engine (probability scoring)
5. Execution Layer (Ollama integration + fallback)
6. RAG System (FAISS + embeddings + retriever)
7. Evaluation Engine (LLM judge async)
8. Learning Engine (Bayesian updater)
9. Metrics Tracker
10. Streamlit Dashboard
11. Security Layer
12. Tests

For each module give me:
- Clean modular Python code
- Proper error handling
- Docstrings on all functions
- How to test the module

IMPORTANT RULES:
- Do NOT overengineer
- Do NOT add features not listed here
- Keep code clean, readable, production-style
- One module at a time
- Confirm you understand before starting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Confirm you have fully understood the architecture.
Then ask me which module to start with.
```

---

# END OF DOCUMENT
