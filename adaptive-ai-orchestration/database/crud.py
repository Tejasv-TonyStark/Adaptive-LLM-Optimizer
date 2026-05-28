# database/crud.py

from sqlalchemy.orm import Session
from sqlalchemy import update, func
from database.models import Query, Evaluation, Probability, AuditLog, Feedback
from datetime import datetime
from typing import Optional


# ──────────────────────────────────────────
# QUERIES
# ──────────────────────────────────────────

def save_query(
    db: Session,
    session_id: str,
    query_text: str,
    intent: str,
    complexity: str,
    strategy: str,
    model_used: str,
    response: str,
    latency_ms: int,
    fallback_used: bool = False,
    # ── Token fields (NEW) ──────────────────
    input_tokens: Optional[int]   = None,
    output_tokens: Optional[int]  = None,
    total_tokens: Optional[int]   = None,
    estimated_cost: Optional[float] = None,
) -> Query:
    """
    Save a new query and its response to the database.
    Token fields are optional — populated from Bedrock metadata or fallback estimation.
    """
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
    db.add(query)
    db.commit()
    db.refresh(query)
    return query


def get_query_by_id(db: Session, query_id: int) -> Query:
    """
    Fetch a single query by its ID.
    Used by the evaluation engine to load query details.
    """
    return db.query(Query).filter(Query.id == query_id).first()


# ──────────────────────────────────────────
# TOKEN / COST AGGREGATES (NEW)
# ──────────────────────────────────────────

def get_token_stats(db: Session) -> dict:
    """
    Aggregate token and cost metrics across all queries.
    Called by /api/metrics to power the dashboard panels.
    """
    result = db.query(
        func.sum(Query.total_tokens).label("total_tokens"),
        func.avg(Query.total_tokens).label("avg_tokens_per_query"),
        func.sum(Query.estimated_cost).label("total_cost"),
    ).one()

    return {
        "total_tokens":        int(result.total_tokens  or 0),
        "avg_tokens_per_query": round(float(result.avg_tokens_per_query or 0), 1),
        "total_estimated_cost": round(float(result.total_cost or 0), 6),
    }


def get_token_stats_by_model(db: Session) -> list[dict]:
    """
    Per-model breakdown of token usage and cost.
    Used for the 'cost per model' and 'tokens per model' dashboard panels.
    """
    rows = db.query(
        Query.model_used,
        func.sum(Query.input_tokens).label("input_tokens"),
        func.sum(Query.output_tokens).label("output_tokens"),
        func.sum(Query.total_tokens).label("total_tokens"),
        func.sum(Query.estimated_cost).label("total_cost"),
        func.avg(Query.total_tokens).label("avg_tokens"),
    ).group_by(Query.model_used).all()

    return [
        {
            "model":        row.model_used,
            "input_tokens":  int(row.input_tokens  or 0),
            "output_tokens": int(row.output_tokens or 0),
            "total_tokens":  int(row.total_tokens  or 0),
            "total_cost":    round(float(row.total_cost or 0), 6),
            "avg_tokens":    round(float(row.avg_tokens or 0), 1),
        }
        for row in rows
    ]


def get_recent_queries(db: Session, limit: int = 20) -> list[Query]:
    """
    Fetch the most recent queries for the dashboard table.
    Includes token/cost columns that are now part of the Query model.
    """
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
    """
    Save the LLM judge's evaluation scores for a query.
    Called asynchronously after the response is sent to the user.
    """
    evaluation = Evaluation(
        query_id      = query_id,
        relevance     = relevance,
        correctness   = correctness,
        completeness  = completeness,
        quality_score = quality_score,
        reasoning     = reasoning
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    return evaluation


# ──────────────────────────────────────────
# PROBABILITIES
# ──────────────────────────────────────────

def get_probability(db: Session, model: str, complexity: str) -> Probability:
    """
    Fetch the current probability row for a model + complexity combination.
    Called by the Decision Engine before every routing decision.
    """
    return db.query(Probability).filter(
        Probability.model      == model,
        Probability.complexity == complexity
    ).first()


def get_all_probabilities(db: Session) -> list[Probability]:
    """
    Fetch all probability rows.
    Used by the Streamlit dashboard to display the full table.
    """
    return db.query(Probability).all()


def update_probability(db: Session, model: str, complexity: str,
                       new_p_quality: float, new_p_latency: float) -> Probability:
    """
    Update p_quality and p_latency for a model + complexity after a query.
    Called by the Learning Engine using the Bayesian update formula.
    Note: p_cost is NOT updated here — it stays fixed as seed value.
    """
    row = get_probability(db, model, complexity)
    if row:
        row.p_quality    = new_p_quality
        row.p_latency    = new_p_latency
        row.sample_count = row.sample_count + 1
        row.last_updated = datetime.utcnow()
        db.commit()
        db.refresh(row)
    return row


# ──────────────────────────────────────────
# AUDIT LOGS
# ──────────────────────────────────────────

def save_audit_log(db: Session, event_type: str,
                   detail: dict, api_key: str) -> AuditLog:
    """
    Save a security/activity audit log entry.
    Called on every API request, error, or fallback event.
    """
    log = AuditLog(
        event_type = event_type,
        detail     = detail,
        api_key    = api_key
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


# ──────────────────────────────────────────
# FEEDBACK
# ──────────────────────────────────────────

def save_feedback(db: Session, query_id: int,
                  rating: int, comment: str = None) -> Feedback:
    """
    Save user feedback (1-5 stars) for a response.
    Called when user submits a rating from the chat UI.
    """
    feedback = Feedback(
        query_id = query_id,
        rating   = rating,
        comment  = comment
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback
