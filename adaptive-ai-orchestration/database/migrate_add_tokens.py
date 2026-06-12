# database/migrate_add_tokens.py
# One-time migration — adds token + cost columns to existing queries table.
# Run once: python -m database.migrate_add_tokens
# Safe to run multiple times — skips columns that already exist.

from sqlalchemy import text
from database.connection import engine


COLUMNS_TO_ADD = [
    ("input_tokens",   "INTEGER"),
    ("output_tokens",  "INTEGER"),
    ("total_tokens",   "INTEGER"),
    ("estimated_cost", "FLOAT"),
]


def column_exists(conn, table: str, column: str) -> bool:
    result = conn.execute(text(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table}'
        AND column_name = '{column}'
    """))
    return result.fetchone() is not None


def run_migration():
    print("\n── Token Column Migration ──\n")

    with engine.connect() as conn:
        for col_name, col_type in COLUMNS_TO_ADD:
            if column_exists(conn, "queries", col_name):
                print(f"⏭️  Skipped  — {col_name} already exists")
            else:
                conn.execute(text(
                    f"ALTER TABLE queries ADD COLUMN {col_name} {col_type}"
                ))
                conn.commit()
                print(f"✅ Added    — {col_name} ({col_type})")

    print("\n✅ Migration complete — queries table updated!")
    print("   New columns: input_tokens, output_tokens, total_tokens, estimated_cost")
    print("   Existing rows will have NULL — that's expected.\n")


if __name__ == "__main__":
    run_migration()
