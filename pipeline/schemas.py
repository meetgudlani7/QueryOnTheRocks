"""
Pipeline Schemas

Data models for the processing pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class Evidence(BaseModel):
    """Represents a piece of evidence from retrieval."""
    passage: str
    score: float
    source: str  # "qdrant", "bm25", or "fused"
    document_id: str
    language: str
    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result from retrieval phase."""
    query: str
    evidence: List[Evidence]
    retrieval_latency_ms: float


class AudioRequest(BaseModel):
    """Request for audio processing."""
    audio_data: bytes
    format: str = "audio/wav"
    sample_rate: Optional[int] = None
    language: str = "en"


class AudioResponse(BaseModel):
    """Response from audio processing (STT)."""
    transcript: str
    language: str
    stt_latency_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())


class QueryRequest(BaseModel):
    """Request for query processing."""
    query: str = Field(..., min_length=1, max_length=1000)
    language: str = "en"


class QueryResponse(BaseModel):
    """Response from query processing (full RAG pipeline)."""
    query: str
    answer: str
    evidence: List[str]  # Simplified evidence for API response
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    language: str
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())


class GenerationRequest(BaseModel):
    """Request for LLM generation."""
    prompt: str
    context: List[Evidence]
    model: str = "llama-3.1-70b-versatile"
    max_tokens: int = 512
    temperature: float = 0.7


class GenerationResponse(BaseModel):
    """Response from LLM generation."""
    generated_text: str
    model: str
    generation_latency_ms: float


class GuardrailsResult(BaseModel):
    """Result from guardrails check."""
    passed: bool
    confidence: float
    reason: Optional[str] = None
    issues: List[str] = Field(default_factory=list)


class ProcessingStage(str, Enum):
    """Pipeline processing stages."""
    STT = "stt"
    NORMALIZATION = "normalization"
    EMBEDDING = "embedding"
    QDRANT_SEARCH = "qdrant_search"
    BM25_SEARCH = "bm25_search"
    RRF_FUSION = "rrf_fusion"
    EVIDENCE_GATE = "evidence_gate"
    CONTEXT_BUILDING = "context_building"
    LLM_GENERATION = "llm_generation"
    VALIDATION = "validation"
