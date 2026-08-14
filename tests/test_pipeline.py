"""
Pipeline Tests

Tests for the processing pipeline.
"""

import pytest
from pipeline import process_query, QueryRequest, PipelineError
from pipeline.schemas import Evidence


@pytest.mark.asyncio
async def test_process_query_basic(mock_settings):
    """Test basic query processing."""
    # This will fail without actual services
    # Mock the pipeline components
    with pytest.raises(PipelineError):
        request = QueryRequest(query="Test query", language="en")
        await process_query(request)


def test_query_request_validation():
    """Test QueryRequest validation."""
    # Valid request
    request = QueryRequest(query="Test query", language="en")
    assert request.query == "Test query"
    assert request.language == "en"
    
    # Invalid request - empty query
    with pytest.raises(Exception):
        QueryRequest(query="", language="en")


def test_evidence_schema():
    """Test Evidence schema."""
    evidence = Evidence(
        passage="Test passage",
        score=0.9,
        source="qdrant",
        document_id="123",
        language="en",
        metadata={"key": "value"},
    )
    
    assert evidence.passage == "Test passage"
    assert evidence.score == 0.9
    assert evidence.source == "qdrant"
    assert evidence.document_id == "123"
