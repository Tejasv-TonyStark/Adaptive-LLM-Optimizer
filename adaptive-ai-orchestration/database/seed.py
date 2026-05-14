# database/seed.py

from database.connection import SessionLocal
from database.models import Probability


# Initial probability values from architecture document
# Source: MMLU benchmarks + hardware benchmarks on i7 15GB RAM

INITIAL_PROBABILITIES = [

    # ── TinyLlama (MMLU=25.9%, RAM=2GB, avg=400ms) ──
    {"model": "tinyllama", "complexity": "low",    "p_quality": 0.82, "p_latency": 0.08, "p_cost": 0.13},
    {"model": "tinyllama", "complexity": "medium", "p_quality": 0.46, "p_latency": 0.11, "p_cost": 0.13},
    {"model": "tinyllama", "complexity": "high",   "p_quality": 0.26, "p_latency": 0.15, "p_cost": 0.13},

    # ── Phi3 Mini (MMLU=68.8%, RAM=4GB, avg=1500ms) ──
    {"model": "phi3",      "complexity": "low",    "p_quality": 0.90, "p_latency": 0.25, "p_cost": 0.27},
    {"model": "phi3",      "complexity": "medium", "p_quality": 0.88, "p_latency": 0.38, "p_cost": 0.27},
    {"model": "phi3",      "complexity": "high",   "p_quality": 0.69, "p_latency": 0.63, "p_cost": 0.27},

    # ── Mistral 7B (MMLU=62.5%, RAM=8GB, avg=4000ms) ──
    {"model": "mistral",   "complexity": "low",    "p_quality": 0.90, "p_latency": 0.63, "p_cost": 0.53},
    {"model": "mistral",   "complexity": "medium", "p_quality": 0.83, "p_latency": 0.88, "p_cost": 0.53},
    {"model": "mistral",   "complexity": "high",   "p_quality": 0.63, "p_latency": 1.00, "p_cost": 0.53},
]


def seed_probabilities():
    """
    Insert initial probability values into the database.
    Only runs if the table is empty — never overwrites existing data.
    """
    db = SessionLocal()

    try:
        # Check if already seeded
        existing = db.query(Probability).count()
        if existing > 0:
            print(f"⚠️  Probabilities table already has {existing} rows — skipping seed.")
            return

        # Insert all 9 rows
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
        print("✅ Probabilities table seeded successfully!")
        print(f"   → {len(INITIAL_PROBABILITIES)} rows inserted")
        print("   → TinyLlama: 3 rows")
        print("   → Phi3 Mini: 3 rows")
        print("   → Mistral 7B: 3 rows")

    except Exception as e:
        db.rollback()
        print(f"❌ Seeding failed: {e}")

    finally:
        db.close()


if __name__ == "__main__":
    seed_probabilities()

