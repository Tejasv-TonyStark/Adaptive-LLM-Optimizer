# database/crud.py

from sqlalchemy.orm import Session
from sqlalchemy import update
from database.models import Query, Evaluation, Probability, AuditLog, Feedback
from datetime import datetime


# ──────────────────────────────────────────
# QUERIES
# ──────────────────────────────────────────

def save_query(db: Session, session_id: str, query_text: str, intent: str,
               complexity: str, strategy: str, model_used: str,
               response: str, latency_ms: int, fallback_used: bool = False) -> Query:
    """
    Save a new query and its response to the database.
    Called after every user question is answered.
    """
    query = Query(
        session_id    = session_id,
        query_text    = query_text,
        intent        = intent,
        complexity    = complexity,
        strategy      = strategy,
        model_used    = model_used,
        response      = response,
        latency_ms    = latency_ms,
        fallback_used = fallback_used
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
    Note: p_cost is NEVER updated — it stays fixed forever.
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


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    from database.connection import SessionLocal

    db = SessionLocal()

    # Test: save a dummy query
    q = save_query(
        db          = db,
        session_id  = "test-session-001",
        query_text  = "What is the leave policy?",
        intent      = "specific",
        complexity  = "low",
        strategy    = "rag",
        model_used  = "mistral",
        response    = "You are entitled to 18 days of annual leave.",
        latency_ms  = 4200,
        fallback_used = False
    )
    print(f"✅ Query saved — ID: {q.id}")

    # Test: save a dummy audit log
    save_audit_log(
        db         = db,
        event_type = "request",
        detail     = {"query_id": q.id, "model": "mistral"},
        api_key    = "test-key-123"
    )
    print("✅ Audit log saved")

    db.close()
    print("✅ crud.py working correctly!")