"""
Fusion Module

Handles result fusion using Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import List, Dict, Any
from collections import defaultdict

from pipeline.schemas import Evidence

logger = logging.getLogger(__name__)


class FusionError(Exception):
    """Custom exception for fusion errors."""
    pass


def reciprocal_rank_fusion(
    qdrant_results: List[Evidence],
    bm25_results: List[Evidence],
    k: int = 5,
    constant: int = 60,
) -> List[Evidence]:
    """
    Fuse results from multiple search methods using Reciprocal Rank Fusion.
    
    RRF combines ranked lists from different search methods by assigning
    scores based on the reciprocal of the rank position.
    
    Args:
        qdrant_results: Results from Qdrant (semantic search)
        bm25_results: Results from BM25 (keyword search)
        k: Number of results to return
        constant: RRF constant parameter (default 60)
        
    Returns:
        List of fused Evidence objects, sorted by score
        
    Raises:
        FusionError: If fusion fails
    """
    try:
        # Track scores for each document
        document_scores: Dict[str, float] = defaultdict(float)
        document_data: Dict[str, Evidence] = {}
        
        # Process Qdrant results
        for rank, evidence in enumerate(qdrant_results, 1):
            doc_id = evidence.document_id
            rrf_score = 1.0 / (constant + rank)
            document_scores[doc_id] += rrf_score
            document_data[doc_id] = evidence
        
        # Process BM25 results
        for rank, evidence in enumerate(bm25_results, 1):
            doc_id = evidence.document_id
            rrf_score = 1.0 / (constant + rank)
            document_scores[doc_id] += rrf_score
            # Keep Qdrant evidence if already exists
            if doc_id not in document_data:
                document_data[doc_id] = evidence
        
        # Sort by RRF score and get top k
        sorted_docs = sorted(
            document_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        # Create fused results
        fused_results = []
        for doc_id, score in sorted_docs:
            evidence = document_data[doc_id]
            # Create new Evidence with fused score
            fused_evidence = Evidence(
                passage=evidence.passage,
                score=score,
                source="fused",
                document_id=evidence.document_id,
                language=evidence.language,
                metadata=evidence.metadata,
            )
            fused_results.append(fused_evidence)
        
        logger.debug(f"Fused {len(qdrant_results)} Qdrant + {len(bm25_results)} BM25 -> {len(fused_results)} results")
        
        return fused_results
        
    except Exception as e:
        logger.error(f"RRF fusion failed: {e}", exc_info=True)
        raise FusionError(f"RRF fusion failed: {e}")


def deduplicate_results(
    results: List[Evidence],
    similarity_threshold: float = 0.8,
) -> List[Evidence]:
    """
    Deduplicate results based on passage similarity.
    
    Args:
        results: List of Evidence objects
        similarity_threshold: Threshold for considering duplicates
        
    Returns:
        Deduplicated list of Evidence objects
    """
    try:
        # Simple deduplication by document_id
        seen_ids = set()
        deduplicated = []
        
        for evidence in results:
            if evidence.document_id not in seen_ids:
                seen_ids.add(evidence.document_id)
                deduplicated.append(evidence)
        
        logger.debug(f"Deduplicated {len(results)} -> {len(deduplicated)} results")
        
        return deduplicated
        
    except Exception as e:
        logger.error(f"Deduplication failed: {e}", exc_info=True)
        return results  # Return original on failure
