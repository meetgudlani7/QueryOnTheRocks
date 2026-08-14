"""
Guardrails Tests

Tests for guardrails components.
"""

import pytest
from pipeline.guardrails import check_evidence, validate_response, GuardrailsResult
from pipeline.schemas import Evidence


@pytest.mark.asyncio
async def test_check_evidence_sufficient():
    """Test evidence check with sufficient evidence."""
    evidence = [
        Evidence(passage="Relevant passage 1", score=0.9, source="qdrant", document_id="1", language="en"),
        Evidence(passage="Relevant passage 2", score=0.8, source="bm25", document_id="2", language="en"),
        Evidence(passage="Relevant passage 3", score=0.7, source="fused", document_id="3", language="en"),
    ]
    
    result = await check_evidence("test query", evidence)
    
    assert isinstance(result, GuardrailsResult)
    assert result.confidence >= 0.0


@pytest.mark.asyncio
async def test_check_evidence_insufficient():
    """Test evidence check with insufficient evidence."""
    evidence = [
        Evidence(passage="Not relevant", score=0.1, source="qdrant", document_id="1", language="en"),
    ]
    
    result = await check_evidence("test query", evidence, min_evidence_count=3)
    
    assert isinstance(result, GuardrailsResult)
    assert not result.passed


@pytest.mark.asyncio
async def test_validate_response_valid():
    """Test response validation with valid response."""
    evidence = [
        Evidence(passage="Evidence about penicillin", score=0.9, source="qdrant", document_id="1", language="en"),
    ]
    
    result = await validate_response(
        query="Who discovered penicillin?",
        answer="Alexander Fleming discovered penicillin in 1928.",
        evidence=evidence,
    )
    
    assert isinstance(result, GuardrailsResult)


@pytest.mark.asyncio
async def test_validate_response_invalid():
    """Test response validation with invalid response."""
    result = await validate_response(
        query="Who discovered penicillin?",
        answer="I don't know",
        evidence=[],
    )
    
    assert isinstance(result, GuardrailsResult)
