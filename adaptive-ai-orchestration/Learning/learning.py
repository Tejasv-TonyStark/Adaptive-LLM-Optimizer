# learning/learning_engine.py

from sqlalchemy.orm import Session
from database.crud import get_probability, update_probability


def compute_alpha(sample_count: int) -> float:
    """
    Adaptive learning rate.
    High alpha early on — system learns fast from first queries.
    Low alpha later — system becomes stable and trusts accumulated data.

    Formula: alpha = 1 / (1 + sample_count)

    At sample_count=0  → alpha = 1.0  (fully trust new data)
    At sample_count=9  → alpha = 0.1
    At sample_count=99 → alpha = 0.01
    """
    return 1.0 / (1.0 + sample_count)


def update_model_probabilities(db: Session, model: str,
                                complexity: str, quality_score: float,
                                latency_ms: int) -> dict:
    """
    Updates p_quality and p_latency for a model after a query.
    p_cost is NEVER updated — hardware is fixed.

    Formula:
        alpha = 1 / (1 + sample_count)
        P_new = P_old + alpha * (actual - P_old)

    Args:
        db:            database session
        model:         nova-micro / llama3-8b / haiku
        complexity:    low / medium / high
        quality_score: actual quality from evaluation engine (0.0-1.0)
        latency_ms:    actual response latency in milliseconds

    Returns:
        dict with old and new probability values
    """
    prob = get_probability(db, model, complexity)

    if not prob:
        print(f"⚠️  No probability row found for {model} + {complexity}")
        return {}

    # Current values
    old_p_quality  = prob.p_quality
    old_p_latency  = prob.p_latency
    sample_count   = prob.sample_count

    # Adaptive learning rate
    alpha = compute_alpha(sample_count)

    # Normalize latency to 0.0 - 1.0
    # Using 6000ms as maximum reference point
    MAX_LATENCY_MS  = 6000
    actual_p_latency = min(latency_ms / MAX_LATENCY_MS, 1.0)

    # Bayesian update
    new_p_quality = old_p_quality + alpha * (quality_score   - old_p_quality)
    new_p_latency = old_p_latency + alpha * (actual_p_latency - old_p_latency)

    # Round to 4 decimal places
    new_p_quality = round(new_p_quality, 4)
    new_p_latency = round(new_p_latency, 4)

    # Save to database
    update_probability(db, model, complexity, new_p_quality, new_p_latency)

    return {
        "model":          model,
        "complexity":     complexity,
        "alpha":          round(alpha, 4),
        "sample_count":   sample_count,
        "old_p_quality":  old_p_quality,
        "new_p_quality":  new_p_quality,
        "old_p_latency":  old_p_latency,
        "new_p_latency":  new_p_latency,
    }


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    from database.connection import SessionLocal

    db = SessionLocal()

    print("\n── Learning Engine Test ──\n")

    # Simulate a good quality fast response from nova-micro
    result = update_model_probabilities(
        db            = db,
        model         = "nova-micro",
        complexity    = "low",
        quality_score = 0.95,
        latency_ms    = 1200
    )

    print(f"Model:         {result['model']}")
    print(f"Complexity:    {result['complexity']}")
    print(f"Alpha:         {result['alpha']}")
    print(f"Sample count:  {result['sample_count']}")
    print(f"p_quality:     {result['old_p_quality']} → {result['new_p_quality']}")
    print(f"p_latency:     {result['old_p_latency']} → {result['new_p_latency']}")
    print()

    # Simulate a bad quality slow response from llama3-8b
    result2 = update_model_probabilities(
        db            = db,
        model         = "llama3-8b",
        complexity    = "medium",
        quality_score = 0.30,
        latency_ms    = 5500
    )

    print(f"Model:         {result2['model']}")
    print(f"Complexity:    {result2['complexity']}")
    print(f"Alpha:         {result2['alpha']}")
    print(f"p_quality:     {result2['old_p_quality']} → {result2['new_p_quality']}")
    print(f"p_latency:     {result2['old_p_latency']} → {result2['new_p_latency']}")
    print()

    db.close()
    print("✅ Learning engine working correctly!")