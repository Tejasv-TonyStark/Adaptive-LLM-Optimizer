# backend/main.py

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy import text

from Backend.schemas import (
    ChatRequest, ChatResponse,
    FeedbackRequest, FeedbackResponse,
    HealthResponse, MetricsResponse,
    ProbabilityResponse
)
from Backend.dependencies import verify_api_key, limiter
from database.connection import get_db, engine
from database.models import Query, Evaluation, Probability
from database import crud

import time

# ──────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────

app = FastAPI(
    title="Adaptive AI Orchestration System",
    description="Intelligent query routing across multiple LLM models",
    version="1.0.0"
)

# Attach rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allows frontend to talk to this API
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
    """
    Checks if the system and database are alive.
    No API key required — public endpoint.
    """
    try:
        db.execute( text("SELECT 1"))
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
    Main endpoint — receives user question and returns answer.
    Currently returns placeholder response.
    Real model routing added in Module 3-5.
    """
    start_time = time.time()

    # ── Placeholder response until orchestrator is built ──
    response_text = f"Query received: '{body.query}'. Orchestrator coming in Module 3."
    strategy = "pending"
    model = "pending"

    latency_ms = int((time.time() - start_time) * 1000)

    # Save query to database
    saved_query = crud.save_query(
        db            = db,
        session_id    = body.session_id,
        query_text    = body.query,
        intent        = "unknown",
        complexity    = "unknown",
        strategy      = strategy,
        model_used    = model,
        response      = response_text,
        latency_ms    = latency_ms,
        fallback_used = False
    )

    # Save audit log
    crud.save_audit_log(
        db         = db,
        event_type = "request",
        detail     = {"query_id": saved_query.id, "session_id": body.session_id},
        api_key    = api_key
    )

    return ChatResponse(
        response      = response_text,
        strategy_used = strategy,
        model_used    = model,
        latency_ms    = latency_ms,
        quality_score = 0.0,
        query_id      = saved_query.id
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
    """
    Receives user rating (1-5 stars) for a response.
    """
    crud.save_feedback(
        db       = db,
        query_id = body.query_id,
        rating   = body.rating,
        comment  = body.comment
    )

    return FeedbackResponse(
        success = True,
        message = f"Feedback saved. Thank you for rating this response."
    )


# ──────────────────────────────────────────
# ENDPOINT 4 — METRICS
# ──────────────────────────────────────────

@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Returns system performance statistics.
    Used by the Streamlit dashboard.
    """
    total_queries = db.query(Query).count()

    avg_latency = db.query(func.avg(Query.latency_ms)).scalar() or 0.0
    avg_quality = db.query(func.avg(Evaluation.quality_score)).scalar() or 0.0

    # Strategy breakdown
    strategy_rows = db.query(
        Query.strategy,
        func.count(Query.id)
    ).group_by(Query.strategy).all()

    strategy_breakdown = {row[0]: row[1] for row in strategy_rows}

    # Model breakdown
    model_rows = db.query(
        Query.model_used,
        func.count(Query.id)
    ).group_by(Query.model_used).all()

    model_breakdown = {row[0]: row[1] for row in model_rows}

    return MetricsResponse(
        total_queries        = total_queries,
        average_latency_ms   = round(avg_latency, 2),
        average_quality_score = round(avg_quality, 4),
        strategy_breakdown   = strategy_breakdown,
        model_breakdown      = model_breakdown
    )


# ──────────────────────────────────────────
# ENDPOINT 5 — PROBABILITIES
# ──────────────────────────────────────────

@app.get("/api/probabilities", response_model=list[ProbabilityResponse])
def get_probabilities(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Returns current probability table.
    Shows how the Decision Engine is routing queries.
    """
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