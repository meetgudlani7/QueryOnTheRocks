"""
Query Processing Endpoint

Handles text query processing through the RAG pipeline.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

from pipeline.orchestrator import process_query
from pipeline.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/api/query", tags=["query"])
logger = logging.getLogger(__name__)


class TextQueryRequest(BaseModel):
    """Request model for text queries."""
    query: str
    language: str = "en"
    

class TextQueryResponse(BaseModel):
    """Response model for text queries."""
    query: str
    answer: str
    evidence: list[str]
    confidence: float
    grounded: bool
    latency_ms: float
    language: str
    request_id: str


@router.post("")
async def process_text_query(request: TextQueryRequest):
    """
    Process a text query through the RAG pipeline.
    
    Args:
        request: TextQueryRequest containing query and language
        
    Returns:
        TextQueryResponse: Answer with evidence and metadata
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        # Convert to pipeline request
        pipeline_request = QueryRequest(
            query=request.query,
            language=request.language,
        )
        
        # Process through pipeline
        response = await process_query(pipeline_request)
        
        # Convert to API response
        api_response = TextQueryResponse(
            query=response.query,
            answer=response.answer,
            evidence=response.evidence,
            confidence=response.confidence,
            grounded=response.grounded,
            latency_ms=response.latency_ms,
            language=response.language,
            request_id=response.request_id,
        )
        
        return JSONResponse(content=api_response.model_dump())
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
