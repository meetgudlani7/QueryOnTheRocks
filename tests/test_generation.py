"""
Generation Tests

Tests for generation components.
"""

import pytest
from pipeline.generation import build_context, GenerationError
from pipeline.schemas import Evidence


def test_build_context():
    """Test context building."""
    evidence = [
        Evidence(passage="Evidence 1", score=0.9, source="qdrant", document_id="1", language="en"),
        Evidence(passage="Evidence 2", score=0.8, source="bm25", document_id="2", language="en"),
    ]
    
    context = build_context("Test query", evidence)
    
    assert "Test query" in context
    assert "Evidence 1" in context
    assert "Evidence 2" in context
    assert "Instructions" in context


def test_build_context_empty():
    """Test context building with empty evidence."""
    context = build_context("Test query", [])
    
    assert "Test query" in context
    assert "Context:" in context
