"""Atomic online means with five seed pseudo-observations, not Bayesian inference."""
import math
from sqlalchemy import update, case
from database.models import Probability
PRIOR_WEIGHT = 5
MAX_LATENCY_MS = 6000
def compute_alpha(sample_count):
    return max(0.1, 1.0 / (PRIOR_WEIGHT + sample_count + 1))
def update_model_probabilities(db, model, complexity, quality_score, latency_ms):
    if not math.isfinite(quality_score) or not 0 <= quality_score <= 1 or not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("Invalid quality or latency observation")
    p = Probability
    denominator = p.sample_count + PRIOR_WEIGHT + 1.0
    alpha = case((denominator < 10.0, 1.0 / denominator), else_=0.1)
    result = db.execute(update(p).where(p.model == model, p.complexity == complexity).values(
        p_quality=p.p_quality + (quality_score-p.p_quality)*alpha,
        p_latency=p.p_latency + (min(latency_ms/MAX_LATENCY_MS, 1.0)-p.p_latency)*alpha,
        sample_count=p.sample_count+1
    ))
    # Caller commits this together with the evaluation: no partial learned result.
    return result.rowcount
