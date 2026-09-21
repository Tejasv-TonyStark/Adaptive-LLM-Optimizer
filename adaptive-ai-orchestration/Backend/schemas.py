from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator
class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    session_id: str = Field(min_length=3, max_length=100)
    @field_validator("query", "session_id")
    @classmethod
    def not_blank(cls, value):
        if len(value.strip()) < 3:
            raise ValueError("At least 3 non-whitespace characters required")
        return value.strip()
class ChatResponse(BaseModel):
    response: str
    strategy_used: str
    model_used: str
    selected_model: str
    complexity: str
    latency_ms: int
    model_latency_ms: int
    quality_score: float | None = None
    query_id: int
    fallback_used: bool = False
    abstained: bool = False
    evaluation_status: str
    sources: list[dict] = Field(default_factory=list)
    routing_reasons: list[str] = Field(default_factory=list)
class EvaluationResponse(BaseModel):
    query_id: int
    status: str
    quality_score: float | None = None
    reasoning: str | None = None
class FeedbackRequest(BaseModel):
    query_id: int = Field(gt=0)
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(None, max_length=500)
class FeedbackResponse(BaseModel):
    success: bool
    message: str
class HealthResponse(BaseModel):
    status: str
    database: str
    message: str
class MetricsResponse(BaseModel):
    total_queries: int
    average_latency_ms: float
    average_quality_score: float
    strategy_breakdown: dict
    model_breakdown: dict
    cost_by_stage: dict = Field(default_factory=dict)
    unknown_cost_attempts: int = 0
class ProbabilityResponse(BaseModel):
    model: str
    complexity: str
    p_quality: float
    p_latency: float
    p_cost: float
    sample_count: int
class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    email: str = Field(min_length=5, max_length=100, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    password: str = Field(min_length=8, max_length=72)
    @field_validator("password")
    @classmethod
    def password_bytes(cls, value):
        if len(value.encode()) > 72:
            raise ValueError("Password must be at most 72 UTF-8 bytes")
        return value
class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=72)
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    message: str
class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    email: str
    is_active: bool
