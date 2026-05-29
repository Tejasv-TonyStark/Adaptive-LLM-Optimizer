# database/models.py

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    Text,
    UniqueConstraint,
    CheckConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.connection import Base


# ──────────────────────────────────────────
# TABLE 1 — QUERIES
# Stores every user query + routing metadata
# ──────────────────────────────────────────
class Query(Base):
    __tablename__ = "queries"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    session_id    = Column(String(100), nullable=False)
    query_text    = Column(Text, nullable=False)
    intent        = Column(String(20))          # general / specific / unknown
    complexity    = Column(String(10))          # low / medium / high
    strategy      = Column(String(20))          # fast / reasoning / rag
    model_used    = Column(String(50))          # nova-micro / llama3-8b / haiku
    response      = Column(Text)
    latency_ms    = Column(Integer)
    fallback_used = Column(Boolean, default=False)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    # ── Token tracking (NEW) ──────────────────
    # Captured from Bedrock usage metadata, or estimated via fallback
    input_tokens   = Column(Integer,  nullable=True)   # prompt tokens sent to model
    output_tokens  = Column(Integer,  nullable=True)   # tokens in model's response
    total_tokens   = Column(Integer,  nullable=True)   # input + output
    estimated_cost = Column(Float,    nullable=True)   # USD cost for this query

    # Relationships
    evaluation = relationship(
        "Evaluation",
        back_populates="query",
        uselist=False,
        cascade="all, delete-orphan"
    )

    feedback = relationship(
        "Feedback",
        back_populates="query",
        uselist=False,
        cascade="all, delete-orphan"
    )


# ──────────────────────────────────────────
# TABLE 2 — EVALUATIONS
# LLM judge scoring
# ──────────────────────────────────────────
class Evaluation(Base):
    __tablename__ = "evaluations"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    query_id      = Column(Integer, ForeignKey("queries.id"), nullable=False, unique=True)
    relevance     = Column(Float)
    correctness   = Column(Float)
    completeness  = Column(Float)
    quality_score = Column(Float)
    reasoning     = Column(Text)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    query = relationship("Query", back_populates="evaluation")


# ──────────────────────────────────────────
# TABLE 3 — PROBABILITIES
# Learning/routing probability table
# ──────────────────────────────────────────
class Probability(Base):
    __tablename__ = "probabilities"

    __table_args__ = (
        UniqueConstraint("model", "complexity", name="uq_model_complexity"),
    )

    id           = Column(Integer, primary_key=True, autoincrement=True)
    model        = Column(String(50), nullable=False)
    complexity   = Column(String(10), nullable=False)
    p_quality    = Column(Float, nullable=False)
    p_latency    = Column(Float, nullable=False)
    p_cost       = Column(Float, nullable=False)
    sample_count = Column(Integer, default=0)
    last_updated = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )


# ──────────────────────────────────────────
# TABLE 4 — AUDIT LOGS
# Security + debugging logs
# ──────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False)
    detail     = Column(JSON)
    api_key    = Column(String(100))
    timestamp  = Column(DateTime(timezone=True), server_default=func.now())


# ──────────────────────────────────────────
# TABLE 5 — FEEDBACK
# User response ratings
# ──────────────────────────────────────────
class Feedback(Base):
    __tablename__ = "feedback"

    __table_args__ = (
        CheckConstraint("rating >= 1 AND rating <= 5", name="check_rating_range"),
    )

    id         = Column(Integer, primary_key=True, autoincrement=True)
    query_id   = Column(Integer, ForeignKey("queries.id"), nullable=False, unique=True)
    rating     = Column(Integer, nullable=False)
    comment    = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    query = relationship("Query", back_populates="feedback")
