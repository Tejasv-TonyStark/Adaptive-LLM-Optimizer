"""Bounded, user-owned follow-up context. No model call or invented rewrite."""
import re
from database.models import Query

FOLLOWUP = re.compile(r"\b(it|its|they|them|those|these|above)\b|\bthat[?.!]*$|^(and\b|what about\b|how about\b|why\??$|continue\b)", re.I)
EXPLICIT_TOPIC = re.compile(
    r"^(?:what (?:is|are)|who (?:is|are)|define|explain|describe|summari[sz]e)\s+"
    r"(?!(?:it|its|they|them|their|that|those|these|this|the above)\b)\w+", re.I)

def is_followup(query):
    return bool(FOLLOWUP.search(query) and not EXPLICIT_TOPIC.search(query))

def resolve_query(db, user_id, session_id, query):
    if not is_followup(query):
        return dict(resolved=query, history=[], followup=False, needs_clarification=False)
    rows = (db.query(Query).filter_by(user_id=user_id, session_id=session_id, status="completed")
            .order_by(Query.id.desc()).limit(3).all())
    if not rows:
        return dict(resolved=query, history=[], followup=True, needs_clarification=True)
    history = [dict(question=r.query_text[:1000], answer=(r.response or "")[:1000]) for r in reversed(rows)]
    # Retain the last explicit topic across chained pronoun-only follow-ups.
    previous = (rows[0].routing_details or {}).get("analysis", {}).get("conversation", {})
    topic = previous.get("topic") or next((r.query_text for r in rows if not is_followup(r.query_text)), rows[-1].query_text)
    resolved = f"Previous topic: {topic[:1000]}\nFollow-up question: {query}"
    return dict(resolved=resolved, history=history, topic=topic[:1000], followup=True, needs_clarification=False)
