# Backend/schemas.py

from pydantic import BaseModel, Field
from typing import Optional


# ── CHAT ──────────────────────────────────

class ChatRequest(BaseModel):
    query:      str = Field(..., min_length=3, max_length=1000)
    session_id: str = Field(..., min_length=3, max_length=100)

class ChatResponse(BaseModel):
    response:      str
    strategy_used: str
    model_used:    str
    latency_ms:    int
    quality_score: float
    query_id:      int


# ── FEEDBACK ──────────────────────────────

class FeedbackRequest(BaseModel):
    query_id: int
    rating:   int = Field(..., ge=1, le=5)
    comment:  Optional[str] = Field(None, max_length=500)

class FeedbackResponse(BaseModel):
    success: bool
    message: str


# ── HEALTH / METRICS / PROBABILITIES ──────

class HealthResponse(BaseModel):
    status:   str
    database: str
    message:  str

class MetricsResponse(BaseModel):
    total_queries:         int
    average_latency_ms:    float
    average_quality_score: float
    strategy_breakdown:    dict
    model_breakdown:       dict

class ProbabilityResponse(BaseModel):
    model:        str
    complexity:   str
    p_quality:    float
    p_latency:    float
    p_cost:       float
    sample_count: int


# ── AUTH ──────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email:    str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6, max_length=100)

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str
    message:      str

class UserResponse(BaseModel):
    id:        int
    username:  str
    email:     str
    is_active: bool
