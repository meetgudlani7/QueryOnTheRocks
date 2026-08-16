"""
Query Processing Endpoint

Handles text query processing through the RAG pipeline.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from apps.api.dependencies import require_api_key
from pipeline.orchestrator import process_query, process_query_stream
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


async def _stream_query_events(request: TextQueryRequest):
    """
    Formats pipeline.orchestrator.process_query_stream's events as
    Server-Sent Events. Each event is the same JSON-serializable dict the
    orchestrator yields (see its docstring for the full set of "type"
    values); the frontend is responsible for rendering "token" events as
    they arrive and treating the answer as provisional until a "verified"
    (or "unverified") event closes the stream — see roadmap Phase 21 for
    why post-hoc validation can't happen any earlier than that.
    """
    pipeline_request = QueryRequest(query=request.query, language=request.language)
    async for event in process_query_stream(pipeline_request):
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("", dependencies=[Depends(require_api_key)])
async def process_text_query(request: TextQueryRequest, stream: bool = Query(False)):
    """
    Process a text query through the RAG pipeline.

    Args:
        request: TextQueryRequest containing query and language
        stream: when true, responds with a Server-Sent Events stream of
            the answer instead of a single JSON body (roadmap Phase 21).
            Default false — every existing caller (evaluation/benchmark.py,
            the current frontend, tests) is completely unaffected by this
            parameter's existence.

    Returns:
        TextQueryResponse: Answer with evidence and metadata (default), or
        a text/event-stream response when stream=true.

    Raises:
        HTTPException: If query processing fails
    """
    if stream:
        return StreamingResponse(_stream_query_events(request), media_type="text/event-stream")

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
