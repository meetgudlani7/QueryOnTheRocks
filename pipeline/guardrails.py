"""
Guardrails Module

Handles evidence checking and response validation.
"""

import logging
from typing import List, Optional, Tuple
import re

from .schemas import Evidence, GuardrailsResult
from retrieval import fusion

logger = logging.getLogger(__name__)


class GuardrailsError(Exception):
    """Custom exception for guardrails errors."""
    pass


async def check_evidence(
    query: str,
    evidence: List[Evidence],
    min_confidence: float = 0.5,
    min_evidence_count: int = 3,
) -> GuardrailsResult:
    """
    Check if evidence is sufficient to answer the query.
    
    Args:
        query: User query
        evidence: List of Evidence objects
        min_confidence: Minimum confidence threshold
        min_evidence_count: Minimum number of evidence items required
        
    Returns:
        GuardrailsResult with pass/fail status
    """
    try:
        issues = []
        
        # Check 1: Sufficient evidence count
        if len(evidence) < min_evidence_count:
            issues.append(f"Insufficient evidence: only {len(evidence)} items found (need {min_evidence_count})")
        
        # Check 2: Evidence relevance
        query_lower = query.lower()
        relevant_count = 0
        
        for e in evidence:
            passage_lower = e.passage.lower()
            
            # Simple keyword matching
            query_words = set(query_lower.split())
            passage_words = set(passage_lower.split())
            
            overlap = query_words & passage_words
            if len(overlap) > 0:
                relevant_count += 1
        
        relevance_ratio = relevant_count / len(evidence) if evidence else 0
        
        if relevance_ratio < 0.3:  # At least 30% should be relevant
            issues.append(f"Low relevance: only {relevance_ratio:.1%} of evidence is relevant")
        
        # Check 3: Confidence score
        avg_score = sum(e.score for e in evidence) / len(evidence) if evidence else 0
        normalized_confidence = min(avg_score * 2, 1.0)  # Normalize to 0-1 range
        
        if normalized_confidence < min_confidence:
            issues.append(f"Low confidence: average score {avg_score:.3f} below threshold {min_confidence}")
        
        # Determine pass/fail
        passed = len(issues) == 0
        confidence = normalized_confidence if passed else (1.0 - len(issues) * 0.2)
        
        reason = None
        if not passed:
            reason = "; ".join(issues)
        
        return GuardrailsResult(
            passed=passed,
            confidence=confidence,
            reason=reason,
            issues=issues,
        )
        
    except Exception as e:
        logger.error(f"Evidence gate check failed: {e}", exc_info=True)
        return GuardrailsResult(
            passed=False,
            confidence=0.0,
            reason=f"Guardrails error: {e}",
            issues=[f"Guardrails error: {e}"]
        )


async def validate_response(
    query: str,
    answer: str,
    evidence: List[Evidence],
) -> GuardrailsResult:
    """
    Validate that the response is grounded in evidence.
    
    Args:
        query: User query
        answer: Generated answer
        evidence: List of Evidence objects used
        
    Returns:
        GuardrailsResult with validation status
    """
    try:
        issues = []
        
        # Check 1: Answer is not empty
        if not answer or len(answer.strip()) < 3:
            issues.append("Answer is too short or empty")
        
        # Check 2: Answer mentions evidence
        answer_lower = answer.lower()
        evidence_text = " ".join(e.passage.lower() for e in evidence)
        
        # Simple check: answer should share some words with evidence
        answer_words = set(answer_lower.split())
        evidence_words = set(evidence_text.split())
        
        overlap = answer_words & evidence_words
        if len(overlap) < 2:  # At least 2 overlapping words
            issues.append("Answer does not sufficiently reference evidence")
        
        # Check 3: No unsafe content
        unsafe_patterns = [
            r"hate",
            r"violence",
            r"harass",
            r"abuse",
            r"discriminat",
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, answer_lower):
                issues.append(f"Potentially unsafe content detected: {pattern}")
        
        # Determine pass/fail
        passed = len(issues) == 0
        confidence = 1.0 - len(issues) * 0.3
        
        return GuardrailsResult(
            passed=passed,
            confidence=max(confidence, 0.0),
            reason="; ".join(issues) if issues else "Response validated successfully",
            issues=issues,
        )
        
    except Exception as e:
        logger.error(f"Response validation failed: {e}", exc_info=True)
        return GuardrailsResult(
            passed=False,
            confidence=0.0,
            reason=f"Validation error: {e}",
            issues=[f"Validation error: {e}"]
        )
