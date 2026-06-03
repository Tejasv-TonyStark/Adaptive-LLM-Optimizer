# Backend/main.py

from fastapi import FastAPI, Depends, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from Backend.schemas import (
    ChatRequest, ChatResponse,
    FeedbackRequest, FeedbackResponse,
    HealthResponse, MetricsResponse, ProbabilityResponse,
    RegisterRequest, LoginRequest, TokenResponse, UserResponse
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
from auth.auth_handler import verify_password, create_token, verify_jwt
from auth import user_store

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

ALLOWED_ORIGINS = [
    "http://localhost:8501", "http://127.0.0.1:8501",
    "http://localhost:8000", "http://127.0.0.1:8000",
    "http://localhost:3000", "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


# ──────────────────────────────────────────
# BACKGROUND EVALUATION + LEARNING
# ──────────────────────────────────────────

def run_evaluation(query_id, query, response, context,
                   model_used, complexity, latency_ms):
    from Evaluation.Evaluator import evaluate_response
    from Learning.learning_engine import update_model_probabilities
    from database.connection import SessionLocal

    db = SessionLocal()
    try:
        scores = evaluate_response(query, response, context)
        if scores["success"]:
            crud.save_evaluation(
                db=db, query_id=query_id,
                relevance=scores["relevance"], correctness=scores["correctness"],
                completeness=scores["completeness"], quality_score=scores["quality_score"],
                reasoning=scores["reasoning"]
            )
            print(f"✅ Evaluation saved — query {query_id} quality={scores['quality_score']}")

            result = update_model_probabilities(
                db=db, model=model_used, complexity=complexity,
                quality_score=scores["quality_score"], latency_ms=latency_ms
            )
            if result:
                print(f"✅ Probabilities updated — {model_used}+{complexity} "
                      f"p_quality: {result['old_p_quality']} → {result['new_p_quality']}")
        else:
            print(f"⚠️ Evaluation failed: {scores['reasoning']}")
    except Exception as e:
        print(f"❌ Background evaluation error: {e}")
    finally:
        db.close()


# ──────────────────────────────────────────
# ENDPOINT 1 — HEALTH CHECK (public)
# ──────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except:
        db_status = "disconnected"
    return HealthResponse(status="ok", database=db_status,
                          message="Adaptive AI Orchestration System is running")


# ──────────────────────────────────────────
# ENDPOINT 2 — REGISTER (public)
# ──────────────────────────────────────────

@app.post("/api/auth/register", response_model=UserResponse)
def register(body: RegisterRequest):
    if user_store.get_user_by_username(body.username):
        raise HTTPException(status_code=400, detail="Username already taken.")
    if user_store.get_user_by_email(body.email):
        raise HTTPException(status_code=400, detail="Email already registered.")

    user = user_store.create_user(
        username=body.username, email=body.email, password=body.password
    )
    return UserResponse(id=user.id, username=user.username,
                        email=user.email, is_active=user.is_active)


# ──────────────────────────────────────────
# ENDPOINT 3 — LOGIN (public)
# ──────────────────────────────────────────

@app.post("/api/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    user = user_store.get_user_by_username(body.username)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled.")

    token = create_token(user_id=user.id, username=user.username)
    return TokenResponse(
        access_token=token, token_type="bearer",
        username=user.username, message=f"Welcome back, {user.username}!"
    )


# ──────────────────────────────────────────
# ENDPOINT 4 — GET CURRENT USER (JWT)
# ──────────────────────────────────────────

@app.get("/api/auth/me", response_model=UserResponse)
def get_me(token_data: dict = Depends(verify_jwt)):
    user = user_store.get_user_by_id(int(token_data["sub"]))
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserResponse(id=user.id, username=user.username,
                        email=user.email, is_active=user.is_active)


# ──────────────────────────────────────────
# ENDPOINT 5 — MAIN CHAT (JWT protected)
# ──────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(
    request: Request,
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    token_data: dict = Depends(verify_jwt)
):
    start_time = time.time()

    # ── Step 0: Security Guard ──
    routing_preview = orchestrate(body.query)
    security = inspect_query(body.query,
                              retrieval_needed=routing_preview["retrieval_needed"])
    if not security["safe"]:
        crud.save_audit_log(db=db, event_type="blocked",
                            detail={"reason": security["reason"],
                                    "query": body.query[:200],
                                    "user": token_data.get("username")},
                            api_key=token_data.get("sub", ""))
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

    # ── Step 3: RAG ──
    context = None
    if retrieval_needed:
        result  = retrieve(clean_query)
        context = result["context"] if result["found"] else None

    # ── Step 4: Execute ──
    result        = execute_query(query=clean_query, model=model,
                                  strategy=strategy, context=context)
    response_text = result["response"]
    model_used    = result["model_used"]
    fallback_used = result["fallback_used"]
    latency_ms    = int((time.time() - start_time) * 1000)

    # ── Step 5: Save ──
    saved_query = crud.save_query(
        db=db, session_id=body.session_id, query_text=clean_query,
        intent=intent, complexity=complexity, strategy=execution_strategy,
        model_used=model_used, response=response_text,
        latency_ms=latency_ms, fallback_used=fallback_used
    )

    # ── Step 6: Audit ──
    crud.save_audit_log(
        db=db, event_type="request",
        detail={
            "query_id": saved_query.id, "user": token_data.get("username"),
            "session_id": body.session_id, "intent": intent,
            "complexity": complexity, "model": model_used,
            "strategy": execution_strategy, "retrieval_needed": retrieval_needed,
            "rag_context_found": context is not None,
            "latency_ms": latency_ms, "fallback": fallback_used
        },
        api_key=token_data.get("sub", "")
    )

    # ── Step 7: Background evaluation ──
    background_tasks.add_task(
        run_evaluation, query_id=saved_query.id, query=clean_query,
        response=response_text, context=context,
        model_used=model_used, complexity=complexity, latency_ms=latency_ms
    )

    return ChatResponse(
        response=response_text, strategy_used=execution_strategy,
        model_used=model_used, latency_ms=latency_ms,
        quality_score=0.0, query_id=saved_query.id
    )


# ──────────────────────────────────────────
# ENDPOINT 6 — FEEDBACK (JWT)
# ──────────────────────────────────────────

@app.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback(body: FeedbackRequest, db: Session = Depends(get_db),
                    token_data: dict = Depends(verify_jwt)):
    crud.save_feedback(db=db, query_id=body.query_id,
                       rating=body.rating, comment=body.comment)
    return FeedbackResponse(success=True, message="Feedback saved successfully.")


# ──────────────────────────────────────────
# ENDPOINT 7 — METRICS (JWT)
# ──────────────────────────────────────────

@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics(db: Session = Depends(get_db),
                token_data: dict = Depends(verify_jwt)):
    total_queries = db.query(Query).count()
    avg_latency   = db.query(func.avg(Query.latency_ms)).scalar() or 0.0
    avg_quality   = db.query(func.avg(Evaluation.quality_score)).scalar() or 0.0
    strategy_rows = db.query(Query.strategy, func.count(Query.id)).group_by(Query.strategy).all()
    model_rows    = db.query(Query.model_used, func.count(Query.id)).group_by(Query.model_used).all()

    return MetricsResponse(
        total_queries=total_queries, average_latency_ms=round(avg_latency, 2),
        average_quality_score=round(avg_quality, 4),
        strategy_breakdown={r[0]: r[1] for r in strategy_rows},
        model_breakdown={r[0]: r[1] for r in model_rows}
    )


# ──────────────────────────────────────────
# ENDPOINT 8 — PROBABILITIES (JWT)
# ──────────────────────────────────────────

@app.get("/api/probabilities", response_model=list[ProbabilityResponse])
def get_probabilities(db: Session = Depends(get_db),
                      token_data: dict = Depends(verify_jwt)):
    rows = crud.get_all_probabilities(db)
    return [
        ProbabilityResponse(
            model=r.model, complexity=r.complexity, p_quality=r.p_quality,
            p_latency=r.p_latency, p_cost=r.p_cost, sample_count=r.sample_count
        )
        for r in rows
    ]


# ──────────────────────────────────────────
# RUN SERVER
# ──────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Backend.main:app", host="0.0.0.0", port=8000, reload=True)
