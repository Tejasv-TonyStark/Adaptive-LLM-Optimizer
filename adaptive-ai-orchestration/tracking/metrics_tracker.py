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
    Example: { "nova-micro": { "avg_quality": 0.91, "eval_count": 10 } }
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
            "avg_quality": round(float(row[1]), 4),
            "eval_count":  row[2]
        }
        for row in rows
    }


# ──────────────────────────────────────────
# 3. AVERAGE LATENCY PER MODEL
# ──────────────────────────────────────────

def get_avg_latency_per_model(db: Session) -> dict:
    """
    Returns average / min / max response time in ms per model.
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
    Returns how often each routing strategy was used (count + %).
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
# 7. TOKEN STATS — GLOBAL (NEW)
# ──────────────────────────────────────────

def get_token_stats_global(db: Session) -> dict:
    """
    System-wide token and cost totals.
    Used for KPI cards on the dashboard.

    Returns:
        total_tokens        — sum of all total_tokens across queries
        avg_tokens_per_query — mean tokens per query
        total_estimated_cost — sum of estimated_cost in USD
    """
    result = db.query(
        func.sum(Query.total_tokens).label("total_tokens"),
        func.avg(Query.total_tokens).label("avg_tokens"),
        func.sum(Query.estimated_cost).label("total_cost"),
    ).one()

    return {
        "total_tokens":         int(result.total_tokens  or 0),
        "avg_tokens_per_query": round(float(result.avg_tokens or 0), 1),
        "total_estimated_cost": round(float(result.total_cost or 0), 6),
    }


# ──────────────────────────────────────────
# 8. TOKEN STATS — PER MODEL (NEW)
# ──────────────────────────────────────────

def get_token_stats_per_model(db: Session) -> list[dict]:
    """
    Per-model token usage and cost breakdown.
    Used for cost-per-model and token-per-model dashboard charts.

    Returns list of:
        model, input_tokens, output_tokens, total_tokens,
        total_cost (USD), avg_tokens_per_query
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


# ──────────────────────────────────────────
# 9. FULL SUMMARY — single call for dashboard
# ──────────────────────────────────────────

def get_full_metrics(db: Session) -> dict:
    """
    Returns all metrics in one call.
    Used by the Streamlit dashboard.
    """
    return {
        "queries_per_model":     get_queries_per_model(db),
        "avg_quality_per_model": get_avg_quality_per_model(db),
        "avg_latency_per_model": get_avg_latency_per_model(db),
        "strategy_breakdown":    get_strategy_breakdown(db),
        "complexity_breakdown":  get_complexity_breakdown(db),
        "probability_table":     get_probability_table(db),
        # Token metrics (NEW)
        "token_stats_global":    get_token_stats_global(db),
        "token_stats_per_model": get_token_stats_per_model(db),
    }


# ──────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    import json

    db = SessionLocal()
    try:
        metrics = get_full_metrics(db)
        print(json.dumps(metrics, indent=2))
    finally:
        db.close()
