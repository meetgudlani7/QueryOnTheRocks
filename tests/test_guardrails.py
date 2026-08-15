"""
Guardrails Tests

Covers the pre-generation evidence gate and post-generation structured
response validation.
"""

import pytest
from pipeline.guardrails import check_evidence, validate_response, GuardrailsResult
from pipeline.schemas import Evidence, GenerationResponse
from config import settings


def _evidence(score: float, doc_id: str = "1", passage: str = "Relevant passage") -> Evidence:
    return Evidence(passage=passage, score=score, source="fused", document_id=doc_id, language="en")


class TestCheckEvidence:
    @pytest.mark.asyncio
    async def test_empty_evidence_fails(self):
        result = await check_evidence("test query", [])
        assert isinstance(result, GuardrailsResult)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_strong_fused_scores_pass(self):
        """Realistic near-best-possible RRF scores (see IMPLEMENTATION_ROADMAP.md
        Phase 8) must actually pass the gate, not just technically not-crash."""
        evidence = [_evidence(0.031, "1"), _evidence(0.030, "2"), _evidence(0.029, "3")]
        result = await check_evidence("test query", evidence, min_evidence_count=1)
        assert result.passed
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_weak_scores_fail(self):
        evidence = [_evidence(0.0005, "1")]
        result = await check_evidence("test query", evidence, min_evidence_count=1)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_insufficient_count_fails(self):
        evidence = [_evidence(0.031, "1")]
        result = await check_evidence("test query", evidence, min_evidence_count=3)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_confidence_never_exceeds_one(self):
        evidence = [_evidence(0.05, "1")]  # above the theoretical per-list max
        result = await check_evidence("test query", evidence, min_evidence_count=1)
        assert result.confidence <= 1.0


class TestValidateResponse:
    @pytest.mark.asyncio
    async def test_grounded_response_with_valid_ids_passes(self):
        gen = GenerationResponse(
            answer="Alexander Fleming discovered penicillin in 1928.",
            evidence_ids=["1"], grounded=True, confidence=0.9,
            model="test-model", generation_latency_ms=10.0,
        )
        evidence = [
            _evidence(0.03, "1", "Alexander Fleming discovered penicillin, an antibiotic, in 1928."),
            _evidence(0.03, "2", "Unrelated passage about something else entirely."),
        ]
        result = await validate_response(gen, evidence=evidence)
        assert result.passed
        assert result.groundedness_similarity is not None
        assert result.groundedness_similarity >= settings.MIN_GROUNDEDNESS_SIMILARITY

    @pytest.mark.asyncio
    async def test_answer_not_grounded_in_cited_passage_fails(self):
        """A citation-ID that's technically valid but whose passage doesn't
        actually support the claim — the case citation-integrity checking
        alone can't catch, which is exactly why the similarity check exists."""
        gen = GenerationResponse(
            answer="Alexander Fleming discovered penicillin in 1928.",
            evidence_ids=["1"], grounded=True, confidence=0.9,
            model="test-model", generation_latency_ms=10.0,
        )
        evidence = [_evidence(0.03, "1", "The stock market closed higher today amid trading optimism.")]
        result = await validate_response(gen, evidence=evidence)
        assert not result.passed
        assert result.groundedness_similarity is not None
        assert result.groundedness_similarity < settings.MIN_GROUNDEDNESS_SIMILARITY

    @pytest.mark.asyncio
    async def test_not_grounded_fails(self):
        gen = GenerationResponse(
            answer="I don't know.", evidence_ids=[], grounded=False, confidence=0.1,
            model="test-model", generation_latency_ms=10.0,
        )
        result = await validate_response(gen, evidence=[_evidence(0.03, "1")])
        assert not result.passed

    @pytest.mark.asyncio
    async def test_empty_answer_fails(self):
        gen = GenerationResponse(
            answer="", evidence_ids=["1"], grounded=True, confidence=0.9,
            model="test-model", generation_latency_ms=10.0,
        )
        result = await validate_response(gen, evidence=[_evidence(0.03, "1")])
        assert not result.passed

    @pytest.mark.asyncio
    async def test_hallucinated_evidence_id_fails(self):
        """Defense in depth: even if generation.py's own filtering somehow
        let a bad id through, validation must still catch it."""
        gen = GenerationResponse(
            answer="Answer.", evidence_ids=["not_really_retrieved"], grounded=True, confidence=0.9,
            model="test-model", generation_latency_ms=10.0,
        )
        evidence = [_evidence(0.03, "1"), _evidence(0.03, "2")]
        result = await validate_response(gen, evidence=evidence)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_unsafe_content_fails(self):
        gen = GenerationResponse(
            answer="This promotes hate speech.", evidence_ids=["1"], grounded=True, confidence=0.9,
            model="test-model", generation_latency_ms=10.0,
        )
        result = await validate_response(gen, evidence=[_evidence(0.03, "1", "This promotes hate speech.")])
        assert not result.passed

    @pytest.mark.asyncio
    async def test_no_evidence_ids_fails(self):
        gen = GenerationResponse(
            answer="Answer.", evidence_ids=[], grounded=True, confidence=0.9,
            model="test-model", generation_latency_ms=10.0,
        )
        result = await validate_response(gen, evidence=[_evidence(0.03, "1")])
        assert not result.passed
