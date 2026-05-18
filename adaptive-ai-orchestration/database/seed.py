# database/seed.py

from database.connection import SessionLocal
from database.models import Probability

INITIAL_PROBABILITIES = [

    # ── Nova Micro (fast, cheapest) ──
    {"model": "nova-micro",  "complexity": "low",    "p_quality": 0.70, "p_latency": 0.05, "p_cost": 0.10},
    {"model": "nova-micro",  "complexity": "medium", "p_quality": 0.50, "p_latency": 0.08, "p_cost": 0.10},
    {"model": "nova-micro",  "complexity": "high",   "p_quality": 0.30, "p_latency": 0.10, "p_cost": 0.10},

    # ── Llama 3.1 8B (reasoning, balanced) ──
    {"model": "llama3-8b",   "complexity": "low",    "p_quality": 0.82, "p_latency": 0.20, "p_cost": 0.25},
    {"model": "llama3-8b",   "complexity": "medium", "p_quality": 0.78, "p_latency": 0.35, "p_cost": 0.25},
    {"model": "llama3-8b",   "complexity": "high",   "p_quality": 0.65, "p_latency": 0.55, "p_cost": 0.25},

    # ── Claude 3.5 Haiku (RAG + Judge, most capable) ──
    {"model": "haiku",       "complexity": "low",    "p_quality": 0.92, "p_latency": 0.30, "p_cost": 0.40},
    {"model": "haiku",       "complexity": "medium", "p_quality": 0.90, "p_latency": 0.45, "p_cost": 0.40},
    {"model": "haiku",       "complexity": "high",   "p_quality": 0.85, "p_latency": 0.65, "p_cost": 0.40},
]


def seed_probabilities():
    """
    Insert initial probability values for new Bedrock models.
    Only runs if table is empty.
    """
    db = SessionLocal()
    try:
        existing = db.query(Probability).count()
        if existing > 0:
            print(f"⚠️  Table already has {existing} rows — skipping seed.")
            return

        for row in INITIAL_PROBABILITIES:
            prob = Probability(
                model        = row["model"],
                complexity   = row["complexity"],
                p_quality    = row["p_quality"],
                p_latency    = row["p_latency"],
                p_cost       = row["p_cost"],
                sample_count = 0
            )
            db.add(prob)

        db.commit()
        print("✅ Probabilities seeded successfully!")
        print(f"   → Nova Micro:    3 rows")
        print(f"   → Llama 3.1 8B: 3 rows")
        print(f"   → Claude Haiku: 3 rows")

    except Exception as e:
        db.rollback()
        print(f"❌ Seeding failed: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    seed_probabilities()