# database/migrate_add_tokens.py
#
# Run this ONCE on your existing database to add the new token/cost columns.
# Safe to re-run — uses IF NOT EXISTS logic.
#
# Usage:
#   python -m database.migrate_add_tokens

from database.connection import engine
from sqlalchemy import text

MIGRATIONS = [
    "ALTER TABLE queries ADD COLUMN IF NOT EXISTS input_tokens   INTEGER",
    "ALTER TABLE queries ADD COLUMN IF NOT EXISTS output_tokens  INTEGER",
    "ALTER TABLE queries ADD COLUMN IF NOT EXISTS total_tokens   INTEGER",
    "ALTER TABLE queries ADD COLUMN IF NOT EXISTS estimated_cost REAL",
]

def run():
    with engine.connect() as conn:
        for stmt in MIGRATIONS:
            conn.execute(text(stmt))
            print(f"✅ {stmt}")
        conn.commit()
    print("\n✅ Migration complete — token columns added to queries table.")

if __name__ == "__main__":
    run()
