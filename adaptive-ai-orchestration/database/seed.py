"""Idempotent seed values are assumptions, not measured model performance."""
from database.connection import SessionLocal
from database.models import Probability
INITIAL_PROBABILITIES = [
    dict(model=m, complexity=c, p_quality=q, p_latency=l, p_cost=cost)
    for m, cost, values in [
        ("nova-micro", .10, [(.70,.05),(.50,.08),(.30,.10)]),
        ("llama3-8b", .25, [(.82,.20),(.78,.35),(.65,.55)]),
        ("llama3-70b", .40, [(.92,.30),(.90,.45),(.85,.65)])
    ] for c,(q,l) in zip(("low","medium","high"), values)
]
def seed_probabilities(db=None):
    if db is None:
        with SessionLocal() as session:
            seed_probabilities(session)
            session.commit()
        return
    for values in INITIAL_PROBABILITIES:
        if not db.query(Probability).filter_by(model=values["model"], complexity=values["complexity"]).first():
            db.add(Probability(**values, sample_count=0))
    db.flush()
if __name__ == "__main__":
    seed_probabilities()
    print("Seeded missing model/complexity rows.")
