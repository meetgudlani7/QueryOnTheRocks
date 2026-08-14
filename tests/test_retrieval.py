"""
Retrieval Tests

Tests for retrieval components.
"""

import pytest
from retrieval import QdrantStore, BM25Store, reciprocal_rank_fusion
from pipeline.schemas import Evidence


@pytest.mark.asyncio
async def test_bm25_store_basic():
    """Test basic BM25 store functionality."""
    store = BM25Store()
    
    # Add documents
    docs = [
        {"id": "1", "passage": "The quick brown fox", "language": "en"},
        {"id": "2", "passage": "jumps over the lazy dog", "language": "en"},
    ]
    
    await store.add_documents(docs)
    
    # Search
    results = await store.search("quick brown", k=2)
    
    assert len(results) > 0
    assert all("passage" in r for r in results)
    assert all("score" in r for r in results)


def test_rrf_fusion():
    """Test Reciprocal Rank Fusion."""
    qdrant_results = [
        Evidence(passage="Qdrant result 1", score=0.9, source="qdrant", document_id="1", language="en"),
        Evidence(passage="Qdrant result 2", score=0.8, source="qdrant", document_id="2", language="en"),
    ]
    
    bm25_results = [
        Evidence(passage="BM25 result 1", score=0.7, source="bm25", document_id="1", language="en"),
        Evidence(passage="BM25 result 3", score=0.6, source="bm25", document_id="3", language="en"),
    ]
    
    fused = reciprocal_rank_fusion(qdrant_results, bm25_results, k=2)
    
    assert len(fused) == 2
    assert all(e.source == "fused" for e in fused)
    assert all("document_id" in e.document_id for e in fused)


def test_qdrant_store_init():
    """Test QdrantStore initialization."""
    store = QdrantStore()
    
    assert store.url is not None
    assert store.collection is not None
