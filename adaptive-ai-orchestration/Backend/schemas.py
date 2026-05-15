# backend/schemas.py

from pydantic import BaseModel, Field
from typing import Optional


# ──────────────────────────────────────────
# REQUEST SCHEMAS — what comes IN
# ──────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Shape of every incoming question from the user.
    FastAPI automatically validates this before touching our code.
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The user's question"
    )
    session_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Unique session identifier for conversation tracking"
    )


class FeedbackRequest(BaseModel):
    """
    Shape of user feedback submission.
    """
    query_id: int = Field(..., description="ID of the query being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(None, max_length=500, description="Optional comment")


# ──────────────────────────────────────────
# RESPONSE SCHEMAS — what goes OUT
# ──────────────────────────────────────────

class ChatResponse(BaseModel):
    """
    Shape of every response sent back to the user.
    """
    response: str
    strategy_used: str
    model_used: str
    latency_ms: int
    quality_score: float
    query_id: int


class HealthResponse(BaseModel):
    """
    Shape of health check response.
    """
    status: str
    database: str
    message: str


class FeedbackResponse(BaseModel):
    """
    Shape of feedback submission confirmation.
    """
    success: bool
    message: str


class MetricsResponse(BaseModel):
    """
    Shape of system metrics response.
    """
    total_queries: int
    average_latency_ms: float
    average_quality_score: float
    strategy_breakdown: dict
    model_breakdown: dict


class ProbabilityResponse(BaseModel):
    """
    Shape of a single probability row.
    """
    model: str
    complexity: str
    p_quality: float
    p_latency: float
    p_cost: float
    sample_count: int


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    # Test: create a valid chat request
    req = ChatRequest(query="What is the leave policy?", session_id="test-123")
    print(f"✅ ChatRequest valid: {req}")

    # Test: invalid request — too short
    try:
        bad_req = ChatRequest(query="Hi", session_id="test-123")
    except Exception as e:
        print(f"✅ Validation working — short query rejected")

    # Test: invalid feedback — rating out of range
    try:
        bad_feedback = FeedbackRequest(query_id=1, rating=6)
    except Exception as e:
        print(f"✅ Validation working — rating 6 rejected")

    print("✅ schemas.py working correctly!")