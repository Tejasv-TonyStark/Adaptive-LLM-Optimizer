"""PostgreSQL-backed evaluation jobs. Pending work survives API restarts.

A row lock spans evaluation so concurrent PostgreSQL workers cannot evaluate the
same job simultaneously. Evaluation and learning commit together. A crash after
the remote call can repeat that billed call, but cannot double-apply learning.
SQLite is supported for single-worker tests only.
"""
import argparse
import time
from datetime import datetime, timedelta, timezone
from sqlalchemy import or_
from database.connection import SessionLocal
from database.models import Query, Evaluation, Usage
from Evaluation.Evaluator import evaluate_response
from Learning.learning import update_model_probabilities
from Evaluation.calibration import judge_learning_allowed

MAX_ATTEMPTS = 3
def process_one(session_factory=SessionLocal):
    with session_factory() as db:
        now = datetime.now(timezone.utc)
        row = (db.query(Query).filter(Query.evaluation_status == "pending",
               or_(Query.evaluation_retry_at.is_(None), Query.evaluation_retry_at <= now))
               .order_by(Query.id).with_for_update(skip_locked=True).first())
        if row is None:
            return False
        existing = db.query(Evaluation).filter_by(query_id=row.id).first()
        if existing:
            row.evaluation_status = "completed"
            db.commit()
            return True
        row.evaluation_attempts = (row.evaluation_attempts or 0) + 1
        try:
            resolved = ((row.routing_details or {}).get("analysis", {}).get("conversation", {})
                        .get("resolved", row.query_text))
            scores = evaluate_response(resolved, row.response, row.context)
        except Exception as exc:
            scores = dict(success=False, usage=[], reasoning=type(exc).__name__)
        for item in scores.get("usage", []):
            db.add(Usage(query_id=row.id, **item))
        if scores["success"]:
            db.add(Evaluation(query_id=row.id, **{k: scores[k] for k in (
                "relevance", "correctness", "completeness", "quality_score", "reasoning",
                "hallucination_flags", "retrieval_score", "retrieval_warning")}))
            learn = judge_learning_allowed(row.model_used)
            if learn:
                update_model_probabilities(db, row.model_used, row.complexity,
                                           scores["quality_score"], row.model_latency_ms)
            row.routing_details = {**(row.routing_details or {}), "learning": {
                "applied": learn, "reason": "calibrated judge" if learn else "judge not independently calibrated for this model"}}
            row.evaluation_status = "completed"
        elif row.evaluation_attempts >= MAX_ATTEMPTS:
            row.evaluation_status = "failed"
        else:
            row.evaluation_retry_at = now + timedelta(seconds=10 * 2**row.evaluation_attempts)
        db.commit()
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Process one eligible job and exit")
    args = parser.parse_args()
    while True:
        processed = process_one()
        if args.once:
            break
        if not processed:
            time.sleep(2)
if __name__ == "__main__":
    main()
