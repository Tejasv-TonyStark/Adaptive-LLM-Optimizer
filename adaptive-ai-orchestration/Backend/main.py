# Backend/main.py

from fastapi import FastAPI, Depends, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from typing import Optional

from Backend.schemas import (
    ChatRequest, ChatResponse,
    FeedbackRequest, FeedbackResponse,
    HealthResponse, MetricsResponse,
    ModelTokenStats, ProbabilityResponse
)
from Backend.dependencies import verify_api_key, limiter
from database.connection import get_db
from database.models import Query, Evaluation
from database import crud
from core.orchestrator import orchestrate
from core.decision_engine import select_best_model
from Execution.Execution_layer import execute_query
from RAG.retriever import retrieve
from Security.Security_Guard import inspect_query

import time

# ──────────────────────────────────────────
# MODEL PRICING CONFIG (USD per 1000 tokens)
# Update these when Bedrock pricing changes.
# Keys match model_used values in the DB.
# ──────────────────────────────────────────

MODEL_PRICING = {
    "nova-micro": {"input": 0.000035, "output": 0.00014},   # Amazon Nova Micro
    "llama3-8b":  {"input": 0.00022,  "output": 0.00022},   # Meta Llama 3.1 8B
    "haiku":      {"input": 0.00072,  "output": 0.00072},   # Meta Llama 3.3 70B
    "default":   {"input": 0.0002,  "output": 0.0002},   # fallback for unknown models
}


def estimate_tokens_from_text(text: str) -> int:
    """
    Fallback token estimator when Bedrock usage metadata is unavailable.
    Rule of thumb: 1 token ≈ 4 characters for English text.
    """
    return max(1, len(text) // 4)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate USD cost for a single query using per-model pricing.
    Returns cost rounded to 8 decimal places to avoid float precision loss.
    """
    pricing      = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    input_cost   = (input_tokens  / 1000) * pricing["input"]
    output_cost  = (output_tokens / 1000) * pricing["output"]
    return round(input_cost + output_cost, 8)


def resolve_token_usage(
    result:     dict,
    query_text: str,
    model:      str,
) -> dict:
    """
    Extract token counts from the execution result.
    Priority:
      1. Bedrock usage metadata (accurate)
      2. Fallback estimation from text length (approximate)

    Returns a dict with input_tokens, output_tokens,
    total_tokens, and estimated_cost.
    """
    # ── Try Bedrock metadata first ──────────
    input_tokens  = result.get("input_tokens")
    output_tokens = result.get("output_tokens")

    if input_tokens is None or output_tokens is None:
        # Fallback: estimate from text lengths
        input_tokens  = estimate_tokens_from_text(query_text)
        output_tokens = estimate_tokens_from_text(result.get("response", ""))

    total_tokens   = input_tokens + output_tokens
    estimated_cost = calculate_cost(model, input_tokens, output_tokens)

    return {
        "input_tokens":   input_tokens,
        "output_tokens":  output_tokens,
        "total_tokens":   total_tokens,
        "estimated_cost": estimated_cost,
    }


# ──────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────

app = FastAPI(
    title="Adaptive AI Orchestration System",
    description="Intelligent query routing across multiple LLM models",
    version="2.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["Content-Type", "X-API-Key"],
)


# ──────────────────────────────────────────
# BACKGROUND EVALUATION + LEARNING
# ──────────────────────────────────────────

def run_evaluation(query_id: int, query: str,
                   response: str, context: str,
                   model_used: str, complexity: str,
                   latency_ms: int):
    from Evaluation.Evaluator import evaluate_response
    from Learning.learning import update_model_probabilities
    from database.connection import SessionLocal

    db = SessionLocal()
    try:
        scores = evaluate_response(query, response, context)

        if scores["success"]:
            crud.save_evaluation(
                db            = db,
                query_id      = query_id,
                relevance     = scores["relevance"],
                correctness   = scores["correctness"],
                completeness  = scores["completeness"],
                quality_score = scores["quality_score"],
                reasoning     = scores["reasoning"]
            )
            print(f"✅ Evaluation saved for query {query_id} "
                  f"— quality={scores['quality_score']}")

            update_result = update_model_probabilities(
                db            = db,
                model         = model_used,
                complexity    = complexity,
                quality_score = scores["quality_score"],
                latency_ms    = latency_ms
            )

            if update_result:
                print(f"✅ Probabilities updated for {model_used} + {complexity} "
                      f"— p_quality: {update_result['old_p_quality']} "
                      f"→ {update_result['new_p_quality']}")
        else:
            print(f"⚠️  Evaluation failed for query {query_id}: "
                  f"{scores['reasoning']}")

    except Exception as e:
        print(f"❌ Background evaluation error: {e}")
    finally:
        db.close()


# ──────────────────────────────────────────
# ENDPOINT 1 — HEALTH CHECK
# ──────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status   = "ok",
        database = db_status,
        message  = "Adaptive AI Orchestration System is running"
    )


# ──────────────────────────────────────────
# ENDPOINT 2 — MAIN CHAT
# ──────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(
    request: Request,
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    start_time = time.time()

    # ── Step 0: Security Guard ──
    routing_preview   = orchestrate(body.query)
    retrieval_preview = routing_preview["retrieval_needed"]

    security = inspect_query(body.query, retrieval_needed=retrieval_preview)

    if not security["safe"]:
        crud.save_audit_log(
            db         = db,
            event_type = "blocked",
            detail     = {
                "session_id": body.session_id,
                "reason":     security["reason"],
                "query":      body.query[:200]
            },
            api_key    = api_key
        )
        raise HTTPException(status_code=400, detail=security["reason"])

    clean_query = security["clean_query"]

    # ── Step 1: Orchestrate ──
    routing            = orchestrate(clean_query)
    intent             = routing["intent"]
    complexity         = routing["complexity"]
    strategy           = routing["strategy"]
    execution_strategy = routing["execution_strategy"]
    retrieval_needed   = routing["retrieval_needed"]

    # ── Step 2: Decision Engine ──
    decision = select_best_model(db, complexity)
    model    = decision["selected_model"]

    # ── Step 3: RAG retrieval ──
    context = None
    if retrieval_needed:
        retrieval_result = retrieve(clean_query)
        context = retrieval_result["context"] \
                  if retrieval_result["found"] else None

    # ── Step 4: Execute ──
    result = execute_query(
        query    = clean_query,
        model    = model,
        strategy = strategy,
        context  = context
    )

    response_text = result["response"]
    model_used    = result["model_used"]
    fallback_used = result["fallback_used"]
    latency_ms    = int((time.time() - start_time) * 1000)

    # ── Step 4b: Resolve token usage (NEW) ──
    token_info = resolve_token_usage(result, clean_query, model_used)

    # ── Step 5: Save ──
    saved_query = crud.save_query(
        db             = db,
        session_id     = body.session_id,
        query_text     = clean_query,
        intent         = intent,
        complexity     = complexity,
        strategy       = execution_strategy,
        model_used     = model_used,
        response       = response_text,
        latency_ms     = latency_ms,
        fallback_used  = fallback_used,
        # Token fields passed through
        input_tokens   = token_info["input_tokens"],
        output_tokens  = token_info["output_tokens"],
        total_tokens   = token_info["total_tokens"],
        estimated_cost = token_info["estimated_cost"],
    )

    # ── Step 6: Audit log ──
    crud.save_audit_log(
        db         = db,
        event_type = "request",
        detail     = {
            "query_id":          saved_query.id,
            "session_id":        body.session_id,
            "intent":            intent,
            "complexity":        complexity,
            "model":             model_used,
            "strategy":          execution_strategy,
            "retrieval_needed":  retrieval_needed,
            "rag_context_found": context is not None,
            "latency_ms":        latency_ms,
            "fallback":          fallback_used,
            # Token info in audit log too
            "total_tokens":      token_info["total_tokens"],
            "estimated_cost":    token_info["estimated_cost"],
        },
        api_key    = api_key
    )

    # ── Step 7: Background evaluation ──
    background_tasks.add_task(
        run_evaluation,
        query_id   = saved_query.id,
        query      = clean_query,
        response   = response_text,
        context    = context,
        model_used = model_used,
        complexity = complexity,
        latency_ms = latency_ms
    )

    return ChatResponse(
        response        = response_text,
        strategy_used   = execution_strategy,
        model_used      = model_used,
        latency_ms      = latency_ms,
        quality_score   = 0.0,
        query_id        = saved_query.id,
        # Token fields in response (NEW)
        input_tokens    = token_info["input_tokens"],
        output_tokens   = token_info["output_tokens"],
        total_tokens    = token_info["total_tokens"],
        estimated_cost  = token_info["estimated_cost"],
    )


# ──────────────────────────────────────────
# ENDPOINT 3 — FEEDBACK
# ──────────────────────────────────────────

@app.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback(
    body: FeedbackRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    crud.save_feedback(
        db       = db,
        query_id = body.query_id,
        rating   = body.rating,
        comment  = body.comment
    )
    return FeedbackResponse(success=True, message="Feedback saved successfully.")


# ──────────────────────────────────────────
# ENDPOINT 4 — METRICS
# ──────────────────────────────────────────

@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    total_queries = db.query(Query).count()
    avg_latency   = db.query(func.avg(Query.latency_ms)).scalar() or 0.0
    avg_quality   = db.query(func.avg(Evaluation.quality_score)).scalar() or 0.0

    strategy_rows = db.query(
        Query.strategy, func.count(Query.id)
    ).group_by(Query.strategy).all()

    model_rows = db.query(
        Query.model_used, func.count(Query.id)
    ).group_by(Query.model_used).all()

    # ── Token / cost aggregates (NEW) ──────
    token_stats       = crud.get_token_stats(db)
    model_token_stats = crud.get_token_stats_by_model(db)

    return MetricsResponse(
        total_queries         = total_queries,
        average_latency_ms    = round(avg_latency, 2),
        average_quality_score = round(avg_quality, 4),
        strategy_breakdown    = {row[0]: row[1] for row in strategy_rows},
        model_breakdown       = {row[0]: row[1] for row in model_rows},
        # Token aggregates
        total_tokens          = token_stats["total_tokens"],
        avg_tokens_per_query  = token_stats["avg_tokens_per_query"],
        total_estimated_cost  = token_stats["total_estimated_cost"],
        token_stats_by_model  = [
            ModelTokenStats(**row) for row in model_token_stats
        ],
    )


# ──────────────────────────────────────────
# ENDPOINT 5 — PROBABILITIES
# ──────────────────────────────────────────

@app.get("/api/probabilities", response_model=list[ProbabilityResponse])
def get_probabilities(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    rows = crud.get_all_probabilities(db)
    return [
        ProbabilityResponse(
            model        = row.model,
            complexity   = row.complexity,
            p_quality    = row.p_quality,
            p_latency    = row.p_latency,
            p_cost       = row.p_cost,
            sample_count = row.sample_count
        )
        for row in rows
    ]


# ──────────────────────────────────────────
# RUN SERVER
# ──────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Backend.main:app", host="0.0.0.0", port=8000, reload=True)
