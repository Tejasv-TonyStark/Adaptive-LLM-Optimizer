"""Static baseline and opt-in greedy online-mean router. No Bayesian claims."""
from database.crud import get_probability
from tracking.usage import calculate_cost
import random
from core.config import ROUTING_POLICY, EXPLORATION_RATE, MAX_EXPLORATION_COST_USD
from core.model_health import model_health
MODELS = ["nova-micro", "llama3-8b", "llama3-70b"]
STATIC_MODELS = dict(low=MODELS[0], medium=MODELS[1], high=MODELS[2])
SUITABLE_MODELS = dict(low=MODELS, medium=MODELS[1:], high=MODELS[2:])
COMPLEXITY_WEIGHTS = {
    "low": dict(quality=0.2, latency=0.4, cost=0.4),
    "medium": dict(quality=0.5, latency=0.3, cost=0.2),
    "high": dict(quality=0.8, latency=0.15, cost=0.05),
}
def score_model(p_quality, p_latency, p_cost, complexity, sample_count=0):
    w = COMPLEXITY_WEIGHTS[complexity]
    return round(w["quality"]*p_quality - w["latency"]*p_latency - w["cost"]*p_cost, 6)
def select_best_model(db, complexity, input_tokens=256, output_tokens=256, policy=None,
                      exploration_rate=None, rng=None):
    policy = policy or ROUTING_POLICY
    if complexity not in STATIC_MODELS or policy not in {"static", "adaptive"}:
        raise ValueError("Unknown complexity or routing policy")
    costs = {m: calculate_cost(m, input_tokens, output_tokens) for m in MODELS}
    health = {m: model_health(db, m) for m in MODELS}
    eligible = [m for m in SUITABLE_MODELS[complexity] if not health[m]["circuit_open"]]
    if not eligible:
        raise RuntimeError("No suitable healthy model available")
    candidates = []
    for model in MODELS:
        row = get_probability(db, model, complexity)
        if row:
            normalized_cost = costs[model] / max(max(costs.values()), 1e-12)
            candidates.append(dict(model=model, expected_cost=costs[model],
                sample_count=row.sample_count, eligible=model in eligible, health=health[model],
                routing_score=score_model(row.p_quality, row.p_latency, normalized_cost, complexity)
                    - 0.5 * health[model]["failure_rate"]))
    candidates.sort(key=lambda x: (-x["routing_score"], x["expected_cost"], x["model"]))
    complete = len(candidates) == len(MODELS)
    ranked = [c for c in candidates if c["eligible"]]
    baseline = STATIC_MODELS[complexity]
    winner = ranked[0]["model"] if policy == "adaptive" and complete else (
        baseline if baseline in eligible else eligible[0])
    explored = False
    rate = EXPLORATION_RATE if exploration_rate is None else exploration_rate
    if not 0 <= rate <= 1:
        raise ValueError("exploration_rate must be in [0, 1]")
    rng = rng or random.SystemRandom()
    alternatives = [c for c in ranked if c["model"] != winner and c["expected_cost"] <= MAX_EXPLORATION_COST_USD]
    if policy == "adaptive" and complete and alternatives and rng.random() < rate:
        # Prefer the least observed eligible alternative; randomize ties.
        count = min(c["sample_count"] for c in alternatives)
        winner = rng.choice([c["model"] for c in alternatives if c["sample_count"] == count])
        explored = True
    return dict(selected_model=winner, candidates=candidates, policy=policy,
                exploration=explored, eligible_models=eligible,
                fallback=policy == "adaptive" and not complete,
                reason="bounded exploration" if explored else (
                    "health-adjusted weighted utility" if policy == "adaptive" and complete else "complexity baseline with health checks"))
