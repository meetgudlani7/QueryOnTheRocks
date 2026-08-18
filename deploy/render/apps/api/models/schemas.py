"""
API Schemas

Pydantic models for request and response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class AudioRequest(BaseModel):
    """Request model for audio processing."""
    audio_data: bytes
    format: str = "audio/wav"
    sample_rate: Optional[int] = None
    language: str = "en"


class AudioResponse(BaseModel):
    """Response model for audio processing."""
    transcript: str
    language: str
    stt_latency_ms: float
    timestamp: datetime


class QueryRequest(BaseModel):
    """Request model for text queries."""
    query: str = Field(..., min_length=1, max_length=1000)
    language: str = "en"


class QueryResponse(BaseModel):
    """Response model for text queries."""
    query: str
    answer: str
    evidence: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    language: str
    timestamp: datetime


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    timestamp: datetime
    version: str


class MetricsResponse(BaseModel):
    """Response model for performance metrics."""
    stt_latency_ms: Optional[float] = None
    query_latency_ms: Optional[float] = None
    embedding_latency_ms: Optional[float] = None
    qdrant_latency_ms: Optional[float] = None
    bm25_latency_ms: Optional[float] = None
    rrf_latency_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    total_latency_ms: float
    requests_count: int
