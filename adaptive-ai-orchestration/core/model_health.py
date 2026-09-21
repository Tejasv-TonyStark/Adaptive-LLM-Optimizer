"""Recent persisted provider outcomes; shared across API processes without new tables."""
from datetime import datetime, timedelta, timezone
from database.models import Usage

def model_health(db, model, now=None):
    now = now or datetime.now(timezone.utc)
    rows = (db.query(Usage).filter(Usage.model == model, Usage.stage == "generation",
            Usage.created_at >= now - timedelta(minutes=30))
            .order_by(Usage.id.desc()).limit(20).all())
    # Only infrastructure errors trip the circuit, not answer-validation failures.
    failures = [r.status == "failed" for r in rows]
    latest = rows[0].created_at if rows else None
    if latest is not None and latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    circuit_open = (len(rows) >= 3 and all(failures[:3])
                    and all(r.error not in {"InvalidEvidence", "EmptyResponse", "TruncatedResponse"} for r in rows[:3])
                    and latest > now - timedelta(seconds=60))
    return dict(failure_rate=sum(failures)/len(rows) if rows else 0,
                observations=len(rows), circuit_open=bool(circuit_open))
