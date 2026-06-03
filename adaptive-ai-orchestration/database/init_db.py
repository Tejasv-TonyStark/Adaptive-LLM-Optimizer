# database/init_db.py

from database.connection import engine, Base
from database import models  # imports all table classes including User

Base.metadata.create_all(bind=engine)

print("✅ All tables created successfully in PostgreSQL!")
print("   Tables: queries, evaluations, probabilities, audit_logs, feedback, users")
