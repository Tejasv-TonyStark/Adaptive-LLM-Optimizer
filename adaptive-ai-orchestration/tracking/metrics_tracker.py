# tracking/metrics_tracker.py

from sqlalchemy.orm import Session
from sqlalchemy import func
from database.connection import SessionLocal
from database.models import Query, Evaluation, Probability


# ──────────────────────────────────────────
# 1. TOTAL QUERIES PER MODEL
# ──────────────────────────────────────────

def get_queries_per_model(db: Session) -> dict:
    """
    Returns how many queries each model has handled.
    Example: { "nova-micro": 12, "llama3-8b": 5, "haiku": 3 }
    """
    rows = (
        db.query(Query.model_used, func.count(Query.id))
        .group_by(Query.model_used)
        .all()
    )
    return {row[0]: row[1] for row in rows}


# ──────────────────────────────────────────
# 2. AVERAGE QUALITY PER MODEL
# ──────────────────────────────────────────

def get_avg_quality_per_model(db: Session) -> dict:
    """
    Joins queries + evaluations and returns average quality score per model.
    Example: { "nova-micro": 0.91, "llama3-8b": 0.74 }
    """
    rows = (
        db.query(
            Query.model_used,
            func.avg(Evaluation.quality_score).label("avg_quality"),
            func.count(Evaluation.id).label("eval_count")
        )
        .join(Evaluation, Evaluation.query_id == Query.id)
        .group_by(Query.model_used)
        .all()
    )
    return {
        row[0]: {
            "avg_quality":  round(float(row[1]), 4),
            "eval_count":   row[2]
        }
        for row in rows
    }


# ──────────────────────────────────────────
# 3. AVERAGE LATENCY PER MODEL
# ──────────────────────────────────────────

def get_avg_latency_per_model(db: Session) -> dict:
    """
    Returns average response time in ms per model.
    Example: { "nova-micro": 820, "llama3-8b": 2100 }
    """
    rows = (
        db.query(
            Query.model_used,
            func.avg(Query.latency_ms).label("avg_latency"),
            func.min(Query.latency_ms).label("min_latency"),
            func.max(Query.latency_ms).label("max_latency")
        )
        .group_by(Query.model_used)
        .all()
    )
    return {
        row[0]: {
            "avg_ms": round(float(row[1]), 1),
            "min_ms": row[2],
            "max_ms": row[3]
        }
        for row in rows
    }


# ──────────────────────────────────────────
# 4. STRATEGY USAGE BREAKDOWN
# ──────────────────────────────────────────

def get_strategy_breakdown(db: Session) -> dict:
    """
    Returns how often each routing strategy was used.
    Example: { "direct": 14, "rag_enhanced": 6 }
    """
    rows = (
        db.query(Query.strategy, func.count(Query.id))
        .group_by(Query.strategy)
        .all()
    )
    total = sum(row[1] for row in rows)
    return {
        row[0]: {
            "count":   row[1],
            "percent": round((row[1] / total) * 100, 1) if total > 0 else 0
        }
        for row in rows
    }


# ──────────────────────────────────────────
# 5. CURRENT PROBABILITY TABLE
# ──────────────────────────────────────────

def get_probability_table(db: Session) -> list:
    """
    Returns current routing probability for every model + complexity combo.
    """
    rows = db.query(Probability).order_by(
        Probability.complexity,
        Probability.model
    ).all()

    return [
        {
            "model":        row.model,
            "complexity":   row.complexity,
            "p_quality":    round(row.p_quality, 4),
            "p_latency":    round(row.p_latency, 4),
            "p_cost":       round(row.p_cost, 4),
            "sample_count": row.sample_count
        }
        for row in rows
    ]


# ──────────────────────────────────────────
# 6. COMPLEXITY BREAKDOWN
# ──────────────────────────────────────────

def get_complexity_breakdown(db: Session) -> dict:
    """
    Returns how many queries were low / medium / high complexity.
    """
    rows = (
        db.query(Query.complexity, func.count(Query.id))
        .group_by(Query.complexity)
        .all()
    )
    return {row[0]: row[1] for row in rows}


# ──────────────────────────────────────────
# 7. FULL SUMMARY — single call for dashboard
# ──────────────────────────────────────────

def get_full_metrics(db: Session) -> dict:
    """
    Returns all metrics in one call.
    Used by the Streamlit dashboard (Module 10).
    """
    return {
        "queries_per_model":    get_queries_per_model(db),
        "avg_quality_per_model":get_avg_quality_per_model(db),
        "avg_latency_per_model":get_avg_latency_per_model(db),
        "strategy_breakdown":   get_strategy_breakdown(db),
        "complexity_breakdown": get_complexity_breakdown(db),
        "probability_table":    get_probability_table(db)
    }


# ──────────────────────────────────────────
# QUICK TEST — run directly to verify
# ──────────────────────────────────────────

if __name__ == "__main__":
    import json

    db = SessionLocal()
    try:
        metrics = get_full_metrics(db)
        print(json.dumps(metrics, indent=2))
    finally:
        db.close()