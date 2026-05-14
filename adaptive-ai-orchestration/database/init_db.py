from database.connection import engine, Base
from database import models

Base.metadata.create_all(bind=engine)

print("✅ All tables created successfully in PostgreSQL!")