from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Read database credentials
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

# Validate required env variables
if not all([USER, PASSWORD, HOST, PORT, DBNAME]):
    raise ValueError("One or more database environment variables are missing.")

# Construct database URL
DATABASE_URL = (
    f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"
)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for ORM models
class Base(DeclarativeBase):
    pass


# Dependency/helper for DB session management
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Optional connectivity test
if __name__ == "__main__":
    try:
        with engine.connect() as connection:
            print("✅ Database connection successful!")
    except Exception as e:
        print("❌ Database connection failed:", e)


        