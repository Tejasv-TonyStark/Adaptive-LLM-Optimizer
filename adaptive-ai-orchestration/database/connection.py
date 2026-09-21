"""PostgreSQL in deployment; explicit SQLite URL supported for offline tests."""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker, DeclarativeBase
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    values = [os.getenv(k) for k in ("user", "password", "host", "port", "dbname")]
    if not all(values):
        raise ValueError("Set DATABASE_URL (or legacy user/password/host/port/dbname variables).")
    DATABASE_URL = URL.create("postgresql+psycopg2", username=values[0], password=values[1],
                             host=values[2], port=int(values[3]), database=values[4],
                             query={"sslmode": "require"})
kwargs = {"pool_pre_ping": True}
if str(DATABASE_URL).startswith("sqlite"):
    kwargs["connect_args"] = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, **kwargs)
if engine.dialect.name == "sqlite":
    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def sqlite_foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
class Base(DeclarativeBase):
    pass
def get_db():
    with SessionLocal() as db:
        yield db
