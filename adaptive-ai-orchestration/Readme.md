# Adaptive AI Orchestration System

Current routing safeguards, setup, verification commands and known limitations
are documented in [HARDENING.md](HARDENING.md). Judge-driven learning now requires
independent calibration; document answers use verified extractive citations.

An intelligent LLM routing system that automatically selects the best AI model for each query based on complexity, intent, and continuously learned probabilities — built with FastAPI, AWS Bedrock, PostgreSQL, and FAISS.

---

## What This System Does

Instead of sending every query to one model, this system analyzes each query and routes it to the most appropriate model:

- **Simple questions** → Nova Micro (fast, cheap)
- **Reasoning questions** → Llama 3.1 8B (balanced)
- **Complex analysis** → Llama 3.3 70B (powerful)
- **Document questions** → RAG pipeline (retrieves from company PDFs first)

After every response, an LLM judge evaluates quality and a Bayesian learning engine updates routing probabilities — so the system improves with every query.

---

## Architecture

```
User Query (HTML Frontend)
         │
         ▼
  Security Guard           ← injection detection, sanitization, RAG guardrails
         │
         ▼
    Orchestrator            ← intent detection + complexity analysis
         │
         ▼
  Decision Engine           ← Bayesian probability routing
         │
    ┌────┴────┐
    │         │
  [RAG]   [Direct]         ← FAISS retrieval if company docs needed
    │         │
    └────┬────┘
         │
         ▼
  AWS Bedrock LLM           ← Nova Micro / Llama 3.1 8B / Llama 3.3 70B
         │
         ▼
   Response → User          ← immediate
         │
    (background)
         │
         ▼
  LLM Judge Evaluation      ← relevance + correctness + completeness + hallucination check
         │
         ▼
  Bayesian Learning Engine  ← updates p_quality + p_latency in DB
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Database | PostgreSQL on Supabase via SQLAlchemy |
| AI Models | AWS Bedrock (us-east-1) |
| Vector DB | FAISS |
| Embeddings | Amazon Titan Text V2 |
| RAG | FAISS + cosine similarity |
| Security | bcrypt, JWT, SHA-256, CORS lockdown |
| Dashboard | Streamlit + Plotly |
| Frontend | HTML/CSS/JS |
| Language | Python 3.13 |

---

## Models

| Internal Name | Bedrock Model ID | Role |
|---|---|---|
| `nova-micro` | `amazon.nova-micro-v1:0` | Fast, low complexity |
| `llama3-8b` | `us.meta.llama3-1-8b-instruct-v1:0` | Reasoning, medium |
| `haiku` | `us.meta.llama3-3-70b-instruct-v1:0` | Complex + LLM judge |

---

## Project Structure

```
adaptive-ai-orchestration/
├── Backend/
│   ├── main.py              # FastAPI app — 8 endpoints
│   ├── schemas.py           # Pydantic request/response models
│   └── dependencies.py      # API key verification + rate limiter
├── auth/
│   └── auth_handler.py      # JWT creation, bcrypt password hashing
├── core/
│   ├── orchestrator.py      # Master routing pipeline
│   ├── intent_detector.py   # General vs company-specific classification
│   ├── complexity_analyzer.py # Low/medium/high classification
│   └── decision_engine.py   # Bayesian probability-based model selection
├── database/
│   ├── models.py            # SQLAlchemy table definitions (6 tables)
│   ├── crud.py              # All DB read/write operations
│   ├── connection.py        # PostgreSQL connection
│   ├── seed.py              # Initial probability values
│   └── init_db.py           # Table creation
├── Execution/
│   ├── Bedrock_client.py    # AWS Bedrock API calls
│   ├── Execution_layer.py   # Model execution + fallback chain
│   └── prompt_builder.py    # Token-efficient prompt templates
├── RAG/
│   ├── document_loader.py   # PDF ingestion
│   ├── embedder.py          # Titan V2 embeddings
│   ├── vector_store.py      # FAISS index management
│   └── retriever.py         # Similarity search + threshold filtering
├── Evaluation/
│   ├── Evaluator.py         # LLM-as-judge scoring
│   └── golden_tests.py      # 12 real Q&A tests from company handbook
├── Learning/
│   └── learning_engine.py   # Bayesian p_quality + p_latency updater
├── security/
│   └── security_guard.py    # Prompt injection + input sanitization
├── tracking/
│   └── metrics_tracker.py   # Aggregated performance metrics
├── Dashboard/
│   └── app.py               # Streamlit dashboard
├── Frontend/
│   ├── index.html           # Chat UI with login page
│   └── logo.png             # Info Services logo
├── tests/
│   └── test_suite.py        # Unit tests for all modules
├── documents/               # Company PDFs for RAG
├── faiss_index/             # FAISS index files
├── .env                     # Environment variables
└── requirements.txt
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo>
cd adaptive-ai-orchestration
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
DATABASE_URL=postgresql://...
JWT_SECRET_KEY=your-secret-key-here
API_KEY_HASHES=sha256-hash-of-your-api-key
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
```

### 3. Create database tables

```bash
python -m database.init_db
```

### 4. Seed probability table

```bash
python -m database.seed
```

### 5. Load company documents into RAG

```bash
python -m RAG.document_loader
```

---

## Running the System

Open 3 terminals simultaneously:

```bash
# Terminal 1 — FastAPI backend
python -m Backend.main

# Terminal 2 — Streamlit dashboard
python -m streamlit run Dashboard/app.py

# Terminal 3 — open Frontend/index.html in browser
```

- **Chat UI:** `Frontend/index.html` (open directly in browser)
- **API Docs:** `http://localhost:8000/docs`
- **Dashboard:** `http://localhost:8501`

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/api/health` | Public | System health check |
| POST | `/api/auth/register` | Public | Register new user |
| POST | `/api/auth/login` | Public | Login, returns JWT token |
| GET | `/api/auth/me` | JWT | Get current user info |
| POST | `/api/chat` | JWT | Send query, get AI response |
| POST | `/api/feedback` | JWT | Submit rating (1-5 stars) |
| GET | `/api/metrics` | JWT | System performance stats |
| GET | `/api/probabilities` | JWT | Current routing probability table |

---

## Security Features

| Feature | Implementation |
|---|---|
| Authentication | JWT tokens (24hr expiry) |
| Password storage | bcrypt hashing |
| API key storage | SHA-256 hash in `.env` |
| Input sanitization | XSS, SQL injection, null byte removal |
| Prompt injection | 30+ regex patterns |
| RAG guardrails | Blocks harmful/out-of-scope queries |
| Rate limiting | 10 requests/minute per IP |
| CORS | Locked to known origins only |
| Audit logging | Every request logged to DB |

---

## Routing Logic

The decision engine selects models using a combined probability score:

```
score = p_quality × p_latency × p_cost
```

After each query, the Bayesian learning engine updates `p_quality` and `p_latency`:

```
alpha = 1 / (1 + sample_count)   ← decreases as more data collected
new_p = (1 - alpha) × old_p + alpha × observed_value
```

This means the system learns from real usage — models that perform well get routed more traffic.

---

## Evaluation Pipeline

Every response is evaluated in the background by an LLM judge (Llama 3.3 70B):

```
quality_score = 0.40 × correctness + 0.35 × relevance + 0.25 × completeness
             - 0.05 × hallucination_count (max penalty 0.20)
```

---

## Running Tests

```bash
# Unit tests (all modules)
python -m tests.test_suite

# Security guard only
python -m Security.Security_Guard

# RAG golden tests (12 real Q&A from company handbook)
python -m Evaluation.golden_tests

# Complexity routing
python -m core.complexity_analyzer

# Orchestrator end-to-end
python -m core.orchestrator
```

---

## Dashboard Features

- Live routing decisions table
- Queries per model (bar chart)
- Complexity breakdown (pie chart)
- Average quality per model
- Average latency per model
- Quality score over time (trend line)
- Strategy usage breakdown
- Probability routing table (colour-coded)
- Auto-refresh every 10 seconds

---

## Built By

**P V G P Tejasv**
AI Engineering Intern — Infoservices Digitech India Private Limited
B.Tech CSE (AI) — Amrita Vishwa Vidyapeetham
