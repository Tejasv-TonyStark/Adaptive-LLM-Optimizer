from contextlib import asynccontextmanager
import os
import logging
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from sqlalchemy import func, text, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from Backend.schemas import (
    ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse, HealthResponse,
    MetricsResponse, ProbabilityResponse, RegisterRequest, LoginRequest,
    TokenResponse, UserResponse, EvaluationResponse)
from Backend.dependencies import limiter, current_user, admin_user
from database.connection import get_db, Base
from database.models import Query, Evaluation, Probability, Feedback, Usage, AuditLog
from auth.auth_handler import verify_password, create_token, signing_key
from auth import user_store
from core.service import process_query, QueryRejected, PipelineUnavailable

@asynccontextmanager
async def lifespan(app):
    signing_key()
    yield

app = FastAPI(title="Adaptive LLM Router", version="3.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.middleware("http")
async def fresh_frontend(request: Request, call_next):
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-store"
    return response

@app.exception_handler(SQLAlchemyError)
async def database_error(request: Request, exc: SQLAlchemyError):
    logging.getLogger(__name__).error("Database request failed: %s", type(exc).__name__)
    return JSONResponse(status_code=503, content={"detail": "Database operation failed. Check the database connection and schema migration."})

@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    logging.getLogger(__name__).error("Unhandled request error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "An internal server error occurred. Check the backend logs and retry."})
origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"])

@app.get("/api/health", response_model=HealthResponse)
def health_check(db: Session=Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        # Validate required tables and columns, not just database connectivity.
        for table in Base.metadata.sorted_tables:
            db.execute(select(table).limit(0))
    except Exception:
        db.rollback()
        raise HTTPException(503, "Database unavailable or schema out of date. Run python -m database.init_db.")
    return HealthResponse(status="ok", database="connected", message="API, database and required schema ready")

@app.post("/api/auth/register", response_model=UserResponse, status_code=201)
@limiter.limit("5/minute")
def register(request: Request, body: RegisterRequest, db: Session=Depends(get_db)):
    reserved = {s.strip().casefold() for s in os.getenv("ADMIN_USERNAMES", "").split(",")}
    if body.username.casefold() in reserved:
        raise HTTPException(409, "Username is reserved.")
    if user_store.get_user_by_username(db, body.username) or user_store.get_user_by_email(db, body.email):
        raise HTTPException(409, "Username or email already registered.")
    try:
        return user_store.create_user(db, body.username, body.email, body.password)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Username or email already registered.")

@app.post("/api/auth/login", response_model=TokenResponse)
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest, db: Session=Depends(get_db)):
    user = user_store.get_user_by_username(db, body.username)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Incorrect username or password.")
    if not user.is_active:
        raise HTTPException(403, "Account is disabled.")
    return TokenResponse(access_token=create_token(user.id, user.username),
                         username=user.username, message="Signed in.")

@app.get("/api/auth/me", response_model=UserResponse)
def get_me(user=Depends(current_user)):
    return user

@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(request: Request, body: ChatRequest, db: Session=Depends(get_db), user=Depends(current_user)):
    try:
        row = process_query(db, user.id, body.query, body.session_id)
    except QueryRejected as exc:
        db.add(AuditLog(event_type="blocked", api_key=str(user.id), detail={"reason": str(exc)}))
        db.commit()
        raise HTTPException(400, str(exc))
    except PipelineUnavailable as exc:
        raise HTTPException(503, {"message": str(exc), "query_id": exc.query_id})
    return ChatResponse(response=row.response, strategy_used=row.strategy, model_used=row.model_used,
        selected_model=row.selected_model, complexity=row.complexity, latency_ms=row.latency_ms,
        model_latency_ms=row.model_latency_ms, query_id=row.id, fallback_used=row.fallback_used,
        abstained=row.status == "abstained", evaluation_status=row.evaluation_status,
        sources=row.sources, routing_reasons=row.routing_details["analysis"]["routing_reasons"])

def owned_query(db, query_id, user_id):
    row = db.query(Query).filter_by(id=query_id, user_id=user_id).first()
    if row is None:
        raise HTTPException(404, "Query not found.")
    return row

@app.get("/api/queries/{query_id}/evaluation", response_model=EvaluationResponse)
def evaluation_status(query_id: int, db: Session=Depends(get_db), user=Depends(current_user)):
    row = owned_query(db, query_id, user.id)
    evaluation = db.query(Evaluation).filter_by(query_id=row.id).first()
    return EvaluationResponse(query_id=row.id, status=row.evaluation_status,
        quality_score=evaluation.quality_score if evaluation else None,
        reasoning=evaluation.reasoning if evaluation else None)

@app.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback(body: FeedbackRequest, db: Session=Depends(get_db), user=Depends(current_user)):
    owned_query(db, body.query_id, user.id)
    if db.query(Feedback).filter_by(query_id=body.query_id).first():
        raise HTTPException(409, "Feedback already submitted.")
    db.add(Feedback(**body.model_dump()))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Feedback already submitted.")
    return FeedbackResponse(success=True, message="Feedback saved for human evaluation; it does not train the router.")

@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics(db: Session=Depends(get_db), user=Depends(admin_user)):
    rows = db.query(Query).filter(Query.status.in_(["completed", "abstained"]))
    total = rows.count()
    latency = rows.with_entities(func.avg(Query.latency_ms)).scalar() or 0
    quality = db.query(func.avg(Evaluation.quality_score)).scalar() or 0
    stage_costs = db.query(Usage.stage, func.sum(Usage.estimated_cost)).group_by(Usage.stage).all()
    return MetricsResponse(total_queries=total, average_latency_ms=round(latency, 2),
        average_quality_score=round(quality, 4),
        strategy_breakdown=dict(rows.with_entities(Query.strategy, func.count(Query.id)).group_by(Query.strategy).all()),
        model_breakdown=dict(rows.with_entities(Query.model_used, func.count(Query.id)).group_by(Query.model_used).all()),
        cost_by_stage={s: float(c or 0) for s,c in stage_costs},
        unknown_cost_attempts=db.query(Usage).filter(Usage.estimated_cost.is_(None)).count())

@app.get("/api/probabilities", response_model=list[ProbabilityResponse])
def get_probabilities(db: Session=Depends(get_db), user=Depends(admin_user)):
    return [ProbabilityResponse(model=p.model, complexity=p.complexity, p_quality=p.p_quality,
        p_latency=p.p_latency, p_cost=p.p_cost, sample_count=p.sample_count) for p in db.query(Probability).all()]

app.mount("/", StaticFiles(directory=Path(__file__).resolve().parents[1]/"Frontend", html=True), name="frontend")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Backend.main:app", host="127.0.0.1", port=8000)
