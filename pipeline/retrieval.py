"""
Retrieval Module

Handles document retrieval using Qdrant and BM25.
"""

import time
import logging
from typing import Any, Dict, List, Tuple, Optional
import asyncio

from .schemas import Evidence, RetrievalResult
from retrieval import qdrant_store, bm25_store, fusion, reranker
from retrieval.reranker import RerankerError
from config import settings

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Custom exception for retrieval errors."""
    pass


def _language_filter(language: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Builds a Qdrant payload filter scoping search to {language, "en"} —
    "en" is always included alongside a non-English language because every
    query_id in the source dataset carries both an English passage and a
    translated one (see ingestion/normalize.py), so excluding English would
    throw away half the genuinely relevant candidates, not just off-language
    noise. Returns None (no filter, i.e. today's behavior) when the feature
    flag is off or no language is known.
    """
    if not settings.RETRIEVAL_LANGUAGE_FILTER or not language:
        return None
    languages = sorted({language, "en"})
    return {"must": [{"key": "language", "match": {"any": languages}}]}


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


async def search_qdrant(query: str, k: int = 20, language: Optional[str] = None) -> Tuple[List[Evidence], float]:
    """
    Search using Qdrant vector database.

    Args:
        query: Query string
        k: Number of results to return
        language: Request/detected language, used to scope the search when
            settings.RETRIEVAL_LANGUAGE_FILTER is on (see _language_filter).
            Ignored entirely when the flag is off — default behavior is
            unchanged.

    Returns:
        Tuple of (results, latency_ms)

    Raises:
        RetrievalError: If Qdrant search fails
    """
    start_time = time.perf_counter()

    try:
        results = await qdrant_store.search(query, k=k, filter=_language_filter(language))
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


async def rerank_evidence(query: str, evidence: List[Evidence], k: int) -> List[Evidence]:
    """
    Re-scores fused evidence with a cross-encoder and returns the top k by
    that score, replacing RRF's positional heuristic with an actual
    (query, passage) relevance judgment for the candidates that made it
    through fusion.

    Fails open: if reranking errors for any reason (model load failure,
    inference error, timeout), logs a warning and returns the original
    fused order's top k unchanged — a broken reranker should degrade
    retrieval quality back to today's behavior, not take down the request.
    This mirrors how Qdrant/BM25 already fail independently without
    failing process_query as a whole.
    """
    if not evidence:
        return []

    try:
        scores = await reranker.rerank(query, [e.passage for e in evidence])
    except RerankerError as e:
        logger.warning(f"Reranking failed, falling back to fusion order: {e}")
        return evidence[:k]

    if len(scores) != len(evidence):
        # Should be structurally impossible (rerank() preserves order/count),
        # but never trust a length mismatch enough to zip the wrong score
        # onto the wrong passage.
        logger.warning(
            f"Reranker returned {len(scores)} scores for {len(evidence)} evidence items, "
            "falling back to fusion order"
        )
        return evidence[:k]

    ranked = sorted(zip(scores, evidence), key=lambda pair: pair[0], reverse=True)
    return [
        Evidence(
            passage=e.passage,
            # Deliberately keep the original RRF-scale score, not the
            # cross-encoder's raw logit. The evidence gate downstream
            # (guardrails.check_evidence) normalizes this value against
            # the theoretical max RRF score to compute confidence — a
            # cross-encoder score is an unbounded, model-specific number
            # (observed range roughly -10 to +11 for this model) on a
            # completely different scale, and substituting it in here
            # once produced a confidence so far outside [0, 1] that
            # QueryResponse's own validation raised instead of returning
            # a graceful refusal. The reranker's score is used only for
            # ordering/selection above; it's preserved in metadata for
            # transparency, never trusted as a stand-in for retrieval
            # confidence.
            score=e.score,
            source="reranked",
            document_id=e.document_id,
            language=e.language,
            metadata={**e.metadata, "rerank_score": score},
        )
        for score, e in ranked[:k]
    ]
