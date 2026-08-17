"""
Pipeline Tests

Tests for the processing pipeline.
"""

import pytest
from config import settings
from pipeline import process_query, QueryRequest
from pipeline.schemas import Evidence


@pytest.mark.asyncio
async def test_process_query_without_groq_degrades_gracefully(monkeypatch):
    """
    Pre-existing bug fixed here: this test originally asserted
    process_query() *raises* PipelineError when GROQ_API_KEY is missing.
    That was never actually true — pipeline/orchestrator.py's resilience
    design (asyncio.gather(..., return_exceptions=True) for retrieval,
    an explicit `except GenerationError` handler around generation) makes
    a hard PipelineError essentially unreachable through this path by
    intent: a missing/failing Groq key degrades to a safe, honest,
    zero-confidence response instead. Verified live before writing this
    fix (see OPTIMIZATION_ROADMAP.md Phase 24) — this now documents and
    locks in the real, intentional behavior instead of a code path the
    architecture no longer takes. The previous `mock_settings` fixture
    didn't actually achieve its intended effect either: patching
    "config.settings" doesn't affect pipeline.generation's own
    already-bound `from config import settings` reference to the same
    object, so it never really isolated this test from live config.
    """
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)

    response = await process_query(QueryRequest(query="Test query", language="en"))

    assert response.confidence == 0.0
    assert response.grounded is False


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
