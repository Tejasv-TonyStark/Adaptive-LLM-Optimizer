# database/models.py

from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    DateTime, JSON, ForeignKey, Text,
    UniqueConstraint, CheckConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.connection import Base


class Query(Base):
    __tablename__ = "queries"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    selected_model = Column(String(50))
    model_latency_ms = Column(Integer)
    status        = Column(String(20), default="completed")
    evaluation_status = Column(String(20), default="skipped", index=True)
    evaluation_attempts = Column(Integer, default=0)
    evaluation_retry_at = Column(DateTime(timezone=True))
    context       = Column(Text)
    sources       = Column(JSON, default=list)
    routing_details = Column(JSON, default=dict)
    session_id    = Column(String(100), nullable=False)
    query_text    = Column(Text, nullable=False)
    intent        = Column(String(20))
    complexity    = Column(String(10))
    strategy      = Column(String(20))
    model_used    = Column(String(50))
    response      = Column(Text)
    latency_ms    = Column(Integer)
    fallback_used = Column(Boolean, default=False)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    # ── Token tracking (nullable — old rows will have NULL) ──
    input_tokens   = Column(Integer,  nullable=True)
    output_tokens  = Column(Integer,  nullable=True)
    total_tokens   = Column(Integer,  nullable=True)
    estimated_cost = Column(Float,    nullable=True)

    evaluation = relationship("Evaluation", back_populates="query",
                               uselist=False, cascade="all, delete-orphan")
    feedback   = relationship("Feedback",   back_populates="query",
                               uselist=False, cascade="all, delete-orphan")


class Evaluation(Base):
    __tablename__ = "evaluations"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    query_id      = Column(Integer, ForeignKey("queries.id"), nullable=False, unique=True)
    relevance     = Column(Float)
    correctness   = Column(Float)
    completeness  = Column(Float)
    quality_score = Column(Float)
    reasoning     = Column(Text)
    hallucination_flags = Column(JSON, default=list)
    retrieval_score = Column(Float)
    retrieval_warning = Column(Boolean)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    query = relationship("Query", back_populates="evaluation")


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
    last_updated = Column(DateTime(timezone=True),
                          server_default=func.now(), onupdate=func.now())


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False)
    detail     = Column(JSON)
    api_key    = Column(String(100))
    timestamp  = Column(DateTime(timezone=True), server_default=func.now())


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


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    username        = Column(String(50),  nullable=False, unique=True)
    email           = Column(String(100), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=False)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())


class Usage(Base):
    """Every provider attempt, including failures with unknown billing."""
    __tablename__ = "usage"
    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False, index=True)
    model = Column(String(50), nullable=False)
    stage = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    estimated_cost = Column(Float)
    estimated = Column(Boolean, default=True)
    latency_ms = Column(Integer)
    error = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
