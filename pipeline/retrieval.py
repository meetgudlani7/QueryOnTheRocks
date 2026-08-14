"""
Retrieval Module

Handles document retrieval using Qdrant and BM25.
"""

import time
import logging
from typing import List, Tuple, Optional
import asyncio

from .schemas import Evidence, RetrievalResult
from retrieval import qdrant_store, bm25_store, fusion

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Custom exception for retrieval errors."""
    pass


def normalize_query(query: str) -> str:
    """
    Normalize query for retrieval.
    
    Args:
        query: Raw query string
        
    Returns:
        Normalized query string
    """
    # Basic normalization
    normalized = query.strip()
    normalized = " ".join(normalized.split())  # Collapse multiple spaces
    return normalized


async def search_qdrant(query: str, k: int = 20) -> Tuple[List[Evidence], float]:
    """
    Search using Qdrant vector database.
    
    Args:
        query: Query string
        k: Number of results to return
        
    Returns:
        Tuple of (results, latency_ms)
        
    Raises:
        RetrievalError: If Qdrant search fails
    """
    start_time = time.perf_counter()
    
    try:
        results = await qdrant_store.search(query, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Convert to Evidence objects
        evidence_list = [
            Evidence(
                passage=result["passage"],
                score=result["score"],
                source="qdrant",
                document_id=result.get("document_id", ""),
                language=result.get("language", "en"),
                metadata=result.get("metadata", {}),
            )
            for result in results
        ]
        
        return evidence_list, latency_ms
        
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}", exc_info=True)
        raise RetrievalError(f"Qdrant search failed: {e}")


async def search_bm25(query: str, k: int = 20) -> Tuple[List[Evidence], float]:
    """
    Search using BM25 keyword retrieval.
    
    Args:
        query: Query string
        k: Number of results to return
        
    Returns:
        Tuple of (results, latency_ms)
        
    Raises:
        RetrievalError: If BM25 search fails
    """
    start_time = time.perf_counter()
    
    try:
        results = await bm25_store.search(query, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Convert to Evidence objects
        evidence_list = [
            Evidence(
                passage=result["passage"],
                score=result["score"],
                source="bm25",
                document_id=result.get("document_id", ""),
                language=result.get("language", "en"),
                metadata=result.get("metadata", {}),
            )
            for result in results
        ]
        
        return evidence_list, latency_ms
        
    except Exception as e:
        logger.error(f"BM25 search failed: {e}", exc_info=True)
        raise RetrievalError(f"BM25 search failed: {e}")


def fuse_results(
    qdrant_results: List[Evidence],
    bm25_results: List[Evidence],
    k: int = 5,
) -> List[Evidence]:
    """
    Fuse results from Qdrant and BM25 using Reciprocal Rank Fusion.
    
    Args:
        qdrant_results: Results from Qdrant
        bm25_results: Results from BM25
        k: Number of fused results to return
        
    Returns:
        List of fused Evidence objects
    """
    try:
        # Use RRF to combine results
        fused = fusion.reciprocal_rank_fusion(
            qdrant_results=qdrant_results,
            bm25_results=bm25_results,
            k=k,
        )
        return fused
    except Exception as e:
        logger.error(f"RRF fusion failed: {e}", exc_info=True)
        # Fallback: return top results from Qdrant
        return qdrant_results[:k]
