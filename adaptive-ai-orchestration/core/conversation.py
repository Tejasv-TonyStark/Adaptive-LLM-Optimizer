"""Bounded, user-owned follow-up context. No model call or invented rewrite."""
import re
from database.models import Query

FOLLOWUP = re.compile(r"\b(it|its|they|them|those|these|above)\b|\bthat[?.!]*$|^(and\b|what about\b|how about\b|why\??$|continue\b)", re.I)

def resolve_query(db, user_id, session_id, query):
    if not FOLLOWUP.search(query):
        return dict(resolved=query, history=[], followup=False, needs_clarification=False)
    rows = (db.query(Query).filter_by(user_id=user_id, session_id=session_id, status="completed")
            .order_by(Query.id.desc()).limit(3).all())
    if not rows:
        return dict(resolved=query, history=[], followup=True, needs_clarification=True)
    history = [dict(question=r.query_text[:1000], answer=(r.response or "")[:1000]) for r in reversed(rows)]
    # Retain the last explicit topic across chained pronoun-only follow-ups.
    topic = next((r.query_text for r in rows if not FOLLOWUP.search(r.query_text)), rows[-1].query_text)
    resolved = f"Previous topic: {topic[:1000]}\nFollow-up question: {query}"
    return dict(resolved=resolved, history=history, followup=True, needs_clarification=False)
