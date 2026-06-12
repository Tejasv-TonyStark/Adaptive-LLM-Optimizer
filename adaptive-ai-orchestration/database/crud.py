# database/crud.py

from sqlalchemy.orm import Session
from database.models import Query, Evaluation, Probability, AuditLog, Feedback, User
from datetime import datetime


# ──────────────────────────────────────────
# QUERIES
# ──────────────────────────────────────────

def save_query(db: Session, session_id: str, query_text: str, intent: str,
               complexity: str, strategy: str, model_used: str,
               response: str, latency_ms: int, fallback_used: bool = False,
               input_tokens: int = None, output_tokens: int = None,
               total_tokens: int = None, estimated_cost: float = None) -> Query:
    query = Query(
        session_id     = session_id,
        query_text     = query_text,
        intent         = intent,
        complexity     = complexity,
        strategy       = strategy,
        model_used     = model_used,
        response       = response,
        latency_ms     = latency_ms,
        fallback_used  = fallback_used,
        input_tokens   = input_tokens,
        output_tokens  = output_tokens,
        total_tokens   = total_tokens,
        estimated_cost = estimated_cost,
    )
    db.add(query); db.commit(); db.refresh(query)
    return query

def get_query_by_id(db: Session, query_id: int) -> Query:
    return db.query(Query).filter(Query.id == query_id).first()

def get_recent_queries(db: Session, limit: int = 20) -> list:
    return (
        db.query(Query)
        .order_by(Query.created_at.desc())
        .limit(limit)
        .all()
    )


# ──────────────────────────────────────────
# EVALUATIONS
# ──────────────────────────────────────────

def save_evaluation(db: Session, query_id: int, relevance: float,
                    correctness: float, completeness: float,
                    quality_score: float, reasoning: str) -> Evaluation:
    evaluation = Evaluation(
        query_id=query_id, relevance=relevance, correctness=correctness,
        completeness=completeness, quality_score=quality_score, reasoning=reasoning
    )
    db.add(evaluation); db.commit(); db.refresh(evaluation)
    return evaluation


# ──────────────────────────────────────────
# PROBABILITIES
# ──────────────────────────────────────────

def get_probability(db: Session, model: str, complexity: str) -> Probability:
    return db.query(Probability).filter(
        Probability.model == model,
        Probability.complexity == complexity
    ).first()

def get_all_probabilities(db: Session) -> list[Probability]:
    return db.query(Probability).all()

def update_probability(db: Session, model: str, complexity: str,
                       new_p_quality: float, new_p_latency: float) -> Probability:
    row = get_probability(db, model, complexity)
    if row:
        row.p_quality    = new_p_quality
        row.p_latency    = new_p_latency
        row.sample_count = row.sample_count + 1
        row.last_updated = datetime.utcnow()
        db.commit(); db.refresh(row)
    return row


# ──────────────────────────────────────────
# TOKEN STATS
# ──────────────────────────────────────────

def get_token_stats(db: Session) -> dict:
    """Global token + cost aggregates."""
    from sqlalchemy import func
    result = db.query(
        func.sum(Query.total_tokens).label("total"),
        func.avg(Query.total_tokens).label("avg"),
        func.sum(Query.estimated_cost).label("cost"),
    ).one()
    return {
        "total_tokens":          int(result.total or 0),
        "avg_tokens_per_query":  round(float(result.avg or 0), 1),
        "total_estimated_cost":  round(float(result.cost or 0), 6),
    }

def get_token_stats_by_model(db: Session) -> list:
    """Per-model token + cost breakdown."""
    from sqlalchemy import func
    rows = db.query(
        Query.model_used,
        func.sum(Query.input_tokens).label("input"),
        func.sum(Query.output_tokens).label("output"),
        func.sum(Query.total_tokens).label("total"),
        func.sum(Query.estimated_cost).label("cost"),
    ).group_by(Query.model_used).all()
    return [
        {
            "model":         row.model_used,
            "input_tokens":  int(row.input  or 0),
            "output_tokens": int(row.output or 0),
            "total_tokens":  int(row.total  or 0),
            "total_cost":    round(float(row.cost or 0), 6),
        }
        for row in rows
    ]


# ──────────────────────────────────────────
# AUDIT LOGS
# ──────────────────────────────────────────

def save_audit_log(db: Session, event_type: str,
                   detail: dict, api_key: str) -> AuditLog:
    log = AuditLog(event_type=event_type, detail=detail, api_key=api_key)
    db.add(log); db.commit(); db.refresh(log)
    return log


# ──────────────────────────────────────────
# FEEDBACK
# ──────────────────────────────────────────

def save_feedback(db: Session, query_id: int,
                  rating: int, comment: str = None) -> Feedback:
    feedback = Feedback(query_id=query_id, rating=rating, comment=comment)
    db.add(feedback); db.commit(); db.refresh(feedback)
    return feedback


# ──────────────────────────────────────────
# USERS — JWT Auth
# ──────────────────────────────────────────

def get_user_by_username(db: Session, username: str) -> User:
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str) -> User:
    return db.query(User).filter(User.email == email).first()

def get_user_by_id(db: Session, user_id: int) -> User:
    return db.query(User).filter(User.id == user_id).first()

def create_user(db: Session, username: str,
                email: str, hashed_password: str) -> User:
    user = User(username=username, email=email,
                hashed_password=hashed_password, is_active=True)
    db.add(user); db.commit(); db.refresh(user)
    return user
