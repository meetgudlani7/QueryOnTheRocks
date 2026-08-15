"""
Guardrails Module

Handles the pre-generation evidence gate and post-generation response
validation (Phase 8) — validation now checks the LLM's structured JSON
output (grounded / evidence_ids / confidence) rather than crude text
overlap.
"""

import logging
import re
from typing import List, Optional

from .schemas import Evidence, GenerationResponse, GuardrailsResult
from config import settings
from retrieval.embeddings import embed_texts, EmbeddingError

logger = logging.getLogger(__name__)


class GuardrailsError(Exception):
    """Custom exception for guardrails errors."""
    pass


async def check_evidence(
    query: str,
    evidence: List[Evidence],
    min_confidence: Optional[float] = None,
    min_evidence_count: int = 1,
) -> GuardrailsResult:
    """
    Pre-generation gate: is there enough evidence to even attempt an answer?

    Args:
        query: user query (kept in the signature for logging/future use)
        evidence: fused Evidence list
        min_confidence: minimum normalized RRF score; defaults to
            settings.MIN_RETRIEVAL_SCORE
        min_evidence_count: minimum number of evidence items required

    Returns:
        GuardrailsResult with pass/fail status
    """
    min_confidence = settings.MIN_RETRIEVAL_SCORE if min_confidence is None else min_confidence

    try:
        issues = []

        if len(evidence) < min_evidence_count:
            issues.append(f"Insufficient evidence: only {len(evidence)} items found (need {min_evidence_count})")

        # Normalize RRF scores against the actual maximum a fused document
        # can reach (rank 1 in both Qdrant and BM25): 2 / (RRF_CONSTANT + 1).
        # An earlier version used a flat "score * 2" heuristic, which for
        # real RRF scores (~0.01-0.03) never got anywhere near a 0.3-0.5
        # threshold — it would have rejected even the best possible matches.
        max_possible_score = 2.0 / (settings.RRF_CONSTANT + 1)
        avg_score = sum(e.score for e in evidence) / len(evidence) if evidence else 0.0
        normalized_confidence = min(avg_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0

        if normalized_confidence < min_confidence:
            issues.append(f"Low confidence: normalized score {normalized_confidence:.3f} below threshold {min_confidence}")

        passed = len(issues) == 0
        confidence = normalized_confidence if passed else (normalized_confidence * 0.5)

        return GuardrailsResult(
            passed=passed,
            confidence=confidence,
            reason="; ".join(issues) if issues else None,
            issues=issues,
        )

    except Exception as e:
        logger.error(f"Evidence gate check failed: {e}", exc_info=True)
        return GuardrailsResult(
            passed=False,
            confidence=0.0,
            reason=f"Guardrails error: {e}",
            issues=[f"Guardrails error: {e}"],
        )


_UNSAFE_PATTERNS = [r"hate", r"violence", r"harass", r"abuse", r"discriminat"]


async def _groundedness_similarity(answer: str, cited_passages: List[str]) -> Optional[float]:
    """
    Cosine similarity between the generated answer and the passages it
    cited — an independent check on the LLM's own "grounded" self-report.
    Citation-ID validation (below) only proves the model cited a real
    passage; it says nothing about whether that passage actually supports
    what the model claimed. Embeddings are already normalized
    (retrieval/embeddings.py), so cosine similarity is just the dot product.

    Returns None (never fails validation) if there's nothing to compare, or
    if the embedding call itself fails — an embedding outage shouldn't take
    down answer validation on top of it.
    """
    if not cited_passages:
        return None
    try:
        vectors = await embed_texts([answer] + cited_passages)
    except EmbeddingError as e:
        logger.warning(f"Groundedness similarity check skipped (embedding failed): {e}")
        return None
    answer_vec, passage_vecs = vectors[0], vectors[1:]
    return max(sum(a * b for a, b in zip(answer_vec, pv)) for pv in passage_vecs)


async def validate_response(
    generation_result: GenerationResponse,
    evidence: List[Evidence],
) -> GuardrailsResult:
    """
    Post-generation gate: validates the LLM's structured output rather than
    doing word-overlap on free text.

    Args:
        generation_result: structured output from generation.generate_answer
        evidence: the fused Evidence actually retrieved for this query —
            used both to catch a hallucinated citation that slipped through
            generation's own filtering (defense in depth, not redundant
            paranoia — this function must be safe to call on its own) and
            to embed the cited passages for the groundedness check below.

    Returns:
        GuardrailsResult with validation status
    """
    try:
        issues = []
        valid_id_set = {e.document_id for e in evidence}
        passage_by_id = {e.document_id: e.passage for e in evidence}

        if not generation_result.answer or len(generation_result.answer.strip()) < 3:
            issues.append("Answer is too short or empty")

        if not generation_result.grounded:
            issues.append("Response is not marked as grounded in evidence")

        if not generation_result.evidence_ids:
            issues.append("No evidence_ids referenced")
        else:
            invalid_ids = [i for i in generation_result.evidence_ids if i not in valid_id_set]
            if invalid_ids:
                issues.append(f"References evidence_ids not in retrieved evidence: {invalid_ids}")

        similarity: Optional[float] = None
        if generation_result.evidence_ids:
            cited_passages = [passage_by_id[i] for i in generation_result.evidence_ids if i in passage_by_id]
            similarity = await _groundedness_similarity(generation_result.answer, cited_passages)
            if similarity is not None and similarity < settings.MIN_GROUNDEDNESS_SIMILARITY:
                issues.append(
                    f"Answer not semantically grounded in cited evidence "
                    f"(similarity={similarity:.3f} < {settings.MIN_GROUNDEDNESS_SIMILARITY})"
                )

        if not (0.0 <= generation_result.confidence <= 1.0):
            issues.append("Confidence out of range")

        answer_lower = generation_result.answer.lower()
        for pattern in _UNSAFE_PATTERNS:
            if re.search(pattern, answer_lower):
                issues.append(f"Potentially unsafe content detected: {pattern}")

        passed = len(issues) == 0

        return GuardrailsResult(
            passed=passed,
            confidence=generation_result.confidence if passed else 0.0,
            reason="; ".join(issues) if issues else "Response validated successfully",
            issues=issues,
            groundedness_similarity=similarity,
        )

    except Exception as e:
        logger.error(f"Response validation failed: {e}", exc_info=True)
        return GuardrailsResult(
            passed=False,
            confidence=0.0,
            reason=f"Validation error: {e}",
            issues=[f"Validation error: {e}"],
        )
