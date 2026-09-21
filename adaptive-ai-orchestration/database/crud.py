"""Routing reads. Request writes are transactionally owned by core.service."""
from database.models import Probability
def get_probability(db, model, complexity):
    return db.query(Probability).filter_by(model=model, complexity=complexity).first()
def get_all_probabilities(db):
    return db.query(Probability).all()
