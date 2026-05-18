# backend/main.py

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from Backend.schemas import (
    ChatRequest, ChatResponse,
    FeedbackRequest, FeedbackResponse,
    HealthResponse, MetricsResponse,
    ProbabilityResponse
)

from Backend.dependencies import verify_api_key, limiter
from database.connection import get_db
from database.models import Query, Evaluation
from database import crud
from core.orchestrator import orchestrate

import time


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ──────────────────────────────────────────
# ENDPOINT 1 — HEALTH CHECK
# ──────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except:
        db_status = "disconnected"

    return HealthResponse(
        status="ok",
        database=db_status,
        message="Adaptive AI Orchestration System is running"
    )


# ──────────────────────────────────────────
# ENDPOINT 2 — MAIN CHAT
# ──────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(
    request: Request,
    body: ChatRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Main endpoint.
    Receives user query.
    Runs orchestration logic.
    Module 5 will replace placeholder execution with real Bedrock inference.
    """

    start_time = time.time()

    # ─────────────────────────────────────
    # ORCHESTRATION
    # ─────────────────────────────────────

    routing = orchestrate(body.query)

    intent = routing["intent"]
    complexity = routing["complexity"]
    strategy = routing["strategy"]
    model = routing["model"]
    retrieval_needed = routing["retrieval_needed"]

    # Placeholder response for now
    response_text = (
        f"Query received.\n"
        f"Intent: {intent}\n"
        f"Complexity: {complexity}\n"
        f"Strategy: {strategy}\n"
        f"Model Selected: {model}\n"
        f"RAG Required: {retrieval_needed}\n"
        f"Execution will be connected in Module 5."
    )

    latency_ms = int((time.time() - start_time) * 1000)

    # ─────────────────────────────────────
    # SAVE QUERY
    # ─────────────────────────────────────

    saved_query = crud.save_query(
        db=db,
        session_id=body.session_id,
        query_text=body.query,
        intent=intent,
        complexity=complexity,
        strategy=strategy,
        model_used=model,
        response=response_text,
        latency_ms=latency_ms,
        fallback_used=False
    )

    # ─────────────────────────────────────
    # AUDIT LOG
    # ─────────────────────────────────────

    crud.save_audit_log(
        db=db,
        event_type="request",
        detail={
            "query_id": saved_query.id,
            "session_id": body.session_id,
            "model": model,
            "strategy": strategy,
            "retrieval_needed": retrieval_needed
        },
        api_key=api_key
    )

    return ChatResponse(
        response=response_text,
        strategy_used=strategy,
        model_used=model,
        latency_ms=latency_ms,
        quality_score=0.0,
        query_id=saved_query.id
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
        db=db,
        query_id=body.query_id,
        rating=body.rating,
        comment=body.comment
    )

    return FeedbackResponse(
        success=True,
        message="Feedback saved successfully."
    )


# ──────────────────────────────────────────
# ENDPOINT 4 — METRICS
# ──────────────────────────────────────────

@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    total_queries = db.query(Query).count()

    avg_latency = db.query(func.avg(Query.latency_ms)).scalar() or 0.0
    avg_quality = db.query(func.avg(Evaluation.quality_score)).scalar() or 0.0

    strategy_rows = db.query(
        Query.strategy,
        func.count(Query.id)
    ).group_by(Query.strategy).all()

    strategy_breakdown = {
        row[0]: row[1] for row in strategy_rows
    }

    model_rows = db.query(
        Query.model_used,
        func.count(Query.id)
    ).group_by(Query.model_used).all()

    model_breakdown = {
        row[0]: row[1] for row in model_rows
    }

    return MetricsResponse(
        total_queries=total_queries,
        average_latency_ms=round(avg_latency, 2),
        average_quality_score=round(avg_quality, 4),
        strategy_breakdown=strategy_breakdown,
        model_breakdown=model_breakdown
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
            model=row.model,
            complexity=row.complexity,
            p_quality=row.p_quality,
            p_latency=row.p_latency,
            p_cost=row.p_cost,
            sample_count=row.sample_count
        )
        for row in rows
    ]


# ──────────────────────────────────────────
# RUN SERVER
# ──────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Backend.main:app", host="0.0.0.0", port=8000, reload=True)