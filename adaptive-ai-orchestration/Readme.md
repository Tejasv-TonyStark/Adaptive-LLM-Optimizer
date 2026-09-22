# Adaptive AI Orchestration - Interview Demo

A local public demonstration of LLM routing and retrieval-augmented generation (RAG). It uses AWS Bedrock and a fictional Sankalpa employee handbook. It is an explainable prototype, not a production deployment.

## What it demonstrates

- General questions use a direct LLM route.
- Sankalpa and handbook questions activate RAG.
- RAG creates Titan embeddings, retrieves from FAISS, reranks candidates, and keeps adjacent PDF chunks together.
- Answers are grounded in verified retrieved text. If model evidence cannot be verified, the app returns an exact relevant source sentence instead of inventing an answer.
- Routing cards show strategy, selected model, latency, sources, and evaluation status.

The handbook is fictional. Do not add confidential documents to this public-demo configuration.

## Architecture

```text
Browser UI -> FastAPI -> validation and routing -> direct LLM or RAG
RAG -> Titan embeddings -> FAISS retrieval -> reranking -> cited response
Streamlit dashboard -> aggregate metrics API
Evaluation worker -> optional model-answer quality scoring
```

## RAG document

The active corpus is `documents/sankalpa_employee_handbook_realistic_simulation.pdf`. Rebuild the FAISS index after adding or replacing a PDF.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
.\venv\Scripts\python.exe -m database.init_db
.\venv\Scripts\python.exe -m database.seed
.\venv\Scripts\python.exe -m RAG.vector_store
```

Configure PostgreSQL and AWS Bedrock credentials in `.env`; see `.env.example`.

## Run

```powershell
# API and chat UI
.\venv\Scripts\python.exe -m Backend.main

# Dashboard
.\venv\Scripts\python.exe -m streamlit run Dashboard/app.py

# Optional quality-evaluation worker
.\venv\Scripts\python.exe -m Evaluation.worker
```

- Chat: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`
- Dashboard: `http://127.0.0.1:8501`

This is public-demo mode: chat, feedback, evaluation status, and dashboard metrics do not require sign-in. Keep it local or behind a trusted network boundary because requests can incur AWS usage charges.

## Demo questions

- `What are the Sankalpa standard working hours?`
- `What is the Sankalpa remote work policy?`
- `How does Sankalpa travel reimbursement work?`
- `What is the Sankalpa resignation notice period?`
- `Explain Python lists.`

## Routing and evaluation

Intent and complexity are transparent rule-based heuristics, not a trained classifier. Sankalpa/handbook questions route to RAG; simple policy lookups are normally low complexity, while explain, compare, and summarize tasks are normally medium complexity.

Set `EVALUATION_SAMPLE_RATE=1.0` if every normal model answer should receive an LLM judge score, and run `Evaluation.worker` to process pending work. Retrieval-only responses are excluded from model-quality scoring because they are source extracts rather than model-generated answers.

## Safety controls

- Query limits, basic sanitization, and rate limiting
- Prompt-injection and credential-request screening
- Restricted CORS origins
- Exact-source citation validation for RAG
- No raw HTML rendering of model output

These controls reduce risk but do not guarantee prevention of all prompt-injection or retrieval-relevance failures.

## Limitations

- RAG relevance is improved by hybrid reranking and adjacent chunks but is not guaranteed.
- The extractive fallback uses lexical matching, not a semantic reranker.
- Reindexing embeds the full corpus; it is not incremental.
- LLM judge scores are not independent human ground truth.
- The dashboard and API are intentionally unauthenticated for this demo.

For production, add authentication and document ACLs, incremental indexing, a learned/cross-encoder reranker, human-labeled evaluation, and operational controls.

## Tests

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_regressions tests.test_hardening -q
node --test tests/frontend.test.cjs
```
