# core/decision_engine.py

from sqlalchemy.orm import Session
from database.crud import get_probability
import math

# ──────────────────────────────────────────
# MODELS IN ROUTING POOL
# ──────────────────────────────────────────

MODELS = ["nova-micro", "llama3-8b", "haiku"]

# ──────────────────────────────────────────
# COMPLEXITY-AWARE DYNAMIC WEIGHTS
# ──────────────────────────────────────────

COMPLEXITY_WEIGHTS = {
    "low": {
        "quality": 0.2,
        "latency": 0.4,
        "cost":    0.4,
    },
    "medium": {
        "quality": 0.5,
        "latency": 0.3,
        "cost":    0.2,
    },
    "high": {
        "quality": 0.8,
        "latency": 0.15,
        "cost":    0.05,
    }
}

CONFIDENCE_FLOOR   = 0.3
CONFIDENCE_CEILING = 1.0
CONFIDENCE_SCALE   = 50


def compute_confidence(sample_count: int) -> float:
    """
    Returns confidence multiplier based on sample count.
    Uses logarithmic scaling.

    At sample_count=0  → confidence = 0.30
    At sample_count=10 → confidence ≈ 0.65
    At sample_count=50 → confidence ≈ 1.00
    """
    if sample_count <= 0:
        return CONFIDENCE_FLOOR

    confidence = CONFIDENCE_FLOOR + (CONFIDENCE_CEILING - CONFIDENCE_FLOOR) * \
                 math.log(1 + sample_count) / math.log(1 + CONFIDENCE_SCALE)

    return min(confidence, CONFIDENCE_CEILING)


def score_model(p_quality: float, p_latency: float, p_cost: float,
                complexity: str, sample_count: int) -> float:
    """
    Calculates routing score using complexity-aware weights
    and confidence scaling.

    Higher score = better choice for this complexity level.
    """
    weights = COMPLEXITY_WEIGHTS[complexity]

    raw_score = (weights["quality"] * p_quality) \
              - (weights["latency"] * p_latency) \
              - (weights["cost"]    * p_cost)

    confidence  = compute_confidence(sample_count)
    final_score = confidence * raw_score

    return round(final_score, 4)


def select_best_model(db: Session, complexity: str) -> dict:
    """
    Reads probability table and selects best model
    using complexity-aware dynamic scoring.

    Args:
        db:         database session
        complexity: low / medium / high

    Returns:
        dict with selected model, scores, all candidates
    """
    candidates = []

    for model in MODELS:
        prob = get_probability(db, model, complexity)

        if not prob:
            print(f"⚠️  No probability row for {model} + {complexity}")
            continue

        routing_score = score_model(
            p_quality    = prob.p_quality,
            p_latency    = prob.p_latency,
            p_cost       = prob.p_cost,
            complexity   = complexity,
            sample_count = prob.sample_count
        )

        confidence = compute_confidence(prob.sample_count)

        candidates.append({
            "model":         model,
            "p_quality":     prob.p_quality,
            "p_latency":     prob.p_latency,
            "p_cost":        prob.p_cost,
            "routing_score": routing_score,
            "confidence":    round(confidence, 2),
            "sample_count":  prob.sample_count
        })

    if not candidates:
        return {
            "selected_model": "haiku",
            "routing_score":  0.0,
            "candidates":     [],
            "fallback":       True,
            "reason":         "no probability data found"
        }

    # Sort by score — tiebreaker: prefer cheaper model
    candidates.sort(
        key=lambda x: (
            round(x["routing_score"], 3),
            -x["p_cost"]
        ),
        reverse=True
    )

    winner = candidates[0]

    return {
        "selected_model": winner["model"],
        "routing_score":  winner["routing_score"],
        "confidence":     winner["confidence"],
        "candidates":     candidates,
        "fallback":       False,
        "reason":         f"complexity={complexity} "
                          f"weights={COMPLEXITY_WEIGHTS[complexity]}"
    }


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    from database.connection import SessionLocal

    db = SessionLocal()

    print("\n── Decision Engine Results ──\n")

    expected_winners = {
        "low":    "nova-micro",
        "medium": "llama3-8b",
        "high":   "haiku"
    }

    all_passed = True

    for complexity in ["low", "medium", "high"]:
        result   = select_best_model(db, complexity)
        expected = expected_winners[complexity]
        passed   = result["selected_model"] == expected
        status   = "✅" if passed else "❌"

        if not passed:
            all_passed = False

        print(f"{status} Complexity: {complexity}")
        print(f"   Expected: {expected}")
        print(f"   Winner:   {result['selected_model']} "
              f"(score={result['routing_score']})")
        print(f"   Weights:  {COMPLEXITY_WEIGHTS[complexity]}")
        print(f"   All candidates:")
        for c in result["candidates"]:
            marker = " ← winner" if c["model"] == result["selected_model"] else ""
            print(f"      {c['model']:12} score={c['routing_score']:7} "
                  f"confidence={c['confidence']}{marker}")
        print()

    db.close()

    if all_passed:
        print("✅ Decision engine working correctly!")
    else:
        print("❌ Some routing decisions need adjustment")