"""Create or upgrade tables explicitly; importing this module never mutates a DB."""
from database.migrate_v3 import migrate
if __name__ == "__main__":
    migrate()
    print("Schema ready. Run python -m database.seed next.")
