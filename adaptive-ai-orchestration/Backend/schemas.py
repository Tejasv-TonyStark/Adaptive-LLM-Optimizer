# Backend/schemas.py

from pydantic import BaseModel, Field
from typing import Optional


# ──────────────────────────────────────────
# REQUEST SCHEMAS — what comes IN
# ──────────────────────────────────────────

class ChatRequest(BaseModel):
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
    query_id: int = Field(..., description="ID of the query being rated")
    rating:   int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment:  Optional[str] = Field(None, max_length=500, description="Optional comment")


# ──────────────────────────────────────────
# RESPONSE SCHEMAS — what goes OUT
# ──────────────────────────────────────────

class ChatResponse(BaseModel):
    response:      str
    strategy_used: str
    model_used:    str
    selected_model: Optional[str] = None
    fallback_used:  bool = False
    complexity:    str
    latency_ms:    int
    quality_score: float
    query_id:      int

    # ── Token fields (NEW) ─────────────────
    input_tokens:   Optional[int]   = None   # prompt tokens sent to model
    output_tokens:  Optional[int]   = None   # tokens in model response
    total_tokens:   Optional[int]   = None   # input + output
    estimated_cost: Optional[float] = None   # USD cost for this query


class HealthResponse(BaseModel):
    status:   str
    database: str
    message:  str


class FeedbackResponse(BaseModel):
    success: bool
    message: str


# ── Per-model token breakdown ─────────────
class ModelTokenStats(BaseModel):
    model:         str
    input_tokens:  int
    output_tokens: int
    total_tokens:  int
    total_cost:    float
    avg_tokens:    float


class MetricsResponse(BaseModel):
    total_queries:         int
    average_latency_ms:    float
    average_quality_score: float
    strategy_breakdown:    dict
    model_breakdown:       dict

    # ── Token / cost aggregates (NEW) ──────
    total_tokens:          int             = 0
    avg_tokens_per_query:  float           = 0.0
    total_estimated_cost:  float           = 0.0
    token_stats_by_model:  list[ModelTokenStats] = Field(default_factory=list)


class ProbabilityResponse(BaseModel):
    model:        str
    complexity:   str
    p_quality:    float
    p_latency:    float
    p_cost:       float
    sample_count: int


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    req = ChatRequest(query="What is the leave policy?", session_id="test-123")
    print(f"✅ ChatRequest valid: {req}")

    try:
        bad_req = ChatRequest(query="Hi", session_id="test-123")
    except Exception:
        print("✅ Validation working — short query rejected")

    try:
        bad_feedback = FeedbackRequest(query_id=1, rating=6)
    except Exception:
        print("✅ Validation working — rating 6 rejected")

    print("✅ schemas.py working correctly!")
