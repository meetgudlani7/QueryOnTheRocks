"""
Guardrails Tests

Covers the pre-generation evidence gate and post-generation structured
response validation.
"""

import httpx
import pytest
from pipeline.guardrails import check_evidence, validate_response, GuardrailsResult, _moderate_content
from pipeline.schemas import Evidence, GenerationResponse
from config import settings
import pipeline.guardrails as guardrails_module


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

    @pytest.mark.asyncio
    async def test_confidence_never_goes_negative(self):
        """
        Regression test for a real crash found while wiring Phase 20
        reranking: a deeply negative evidence.score (e.g. from a
        mis-scaled upstream value) must degrade to confidence=0.0, not a
        negative number — QueryResponse's confidence field is constrained
        to [0.0, 1.0], so a negative value here doesn't just look wrong,
        it crashes response construction on what should be a graceful
        refusal path. Deliberately uses a magnitude (-6.4) in the same
        ballpark as the real incident (a cross-encoder logit average of
        roughly -198 pre-clamp).
        """
        evidence = [_evidence(-6.4, "1")]
        result = await check_evidence("test query", evidence, min_evidence_count=1)
        assert not result.passed
        assert 0.0 <= result.confidence <= 1.0


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

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_unsafe_content_fails(self):
        """
        Marked live (roadmap Phase 24): _moderate_content fails open — with
        no GROQ_API_KEY it returns None ("safe") without ever attempting a
        network call, which would make every other check in this fixture
        pass and flip this test's expected outcome. This is the one test
        in the file whose correctness genuinely depends on a real key, not
        just one that happens to make an extra network call.
        """
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


class _FakeAsyncClient:
    """Minimal async-context-manager stand-in for httpx.AsyncClient, used only
    to test _moderate_content's error-handling/fail-open paths deterministically
    — real semantic accuracy (does it actually distinguish description from
    endorsement) was verified manually against the live Groq API; see
    OPTIMIZATION_ROADMAP.md Phase 23 for that result."""

    def __init__(self, response=None, raise_error=None):
        self._response = response
        self._raise_error = raise_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        if self._raise_error:
            raise self._raise_error
        return self._response


class _FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


class TestModerateContent:
    """
    Phase 23's LLM-based content-safety check, replacing a 5-keyword regex
    that was both trivially bypassable and prone to false-positiving on
    ordinary factual content. These tests cover the deterministic,
    offline-testable fail-open paths; real judgment accuracy was verified
    manually against the live model (see OPTIMIZATION_ROADMAP.md Phase 23)
    — it correctly flagged genuinely unsafe content and actionable-harm
    content, and correctly passed a factual description of a historical
    atrocity that the old regex's "violence" keyword would have risked
    false-positiving on.
    """

    @pytest.mark.asyncio
    async def test_empty_text_returns_none(self):
        assert await _moderate_content("") is None
        assert await _moderate_content("   ") is None

    @pytest.mark.asyncio
    async def test_no_api_key_fails_open(self, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_API_KEY", None)
        assert await _moderate_content("anything") is None

    @pytest.mark.asyncio
    async def test_non_200_response_fails_open(self, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_API_KEY", "fake-key-for-test")
        fake = _FakeAsyncClient(response=_FakeResponse(500, text="internal error"))
        monkeypatch.setattr(guardrails_module.httpx, "AsyncClient", lambda **kw: fake)

        assert await _moderate_content("some answer text") is None

    @pytest.mark.asyncio
    async def test_transport_error_fails_open(self, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_API_KEY", "fake-key-for-test")
        fake = _FakeAsyncClient(raise_error=httpx.TransportError("connection reset"))
        monkeypatch.setattr(guardrails_module.httpx, "AsyncClient", lambda **kw: fake)

        assert await _moderate_content("some answer text") is None

    @pytest.mark.asyncio
    async def test_unparseable_output_fails_open(self, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_API_KEY", "fake-key-for-test")
        response = _FakeResponse(200, json_data={"choices": [{"message": {"content": "not valid json at all"}}]})
        fake = _FakeAsyncClient(response=response)
        monkeypatch.setattr(guardrails_module.httpx, "AsyncClient", lambda **kw: fake)

        assert await _moderate_content("some answer text") is None

    @pytest.mark.asyncio
    async def test_flagged_response_returns_reason(self, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_API_KEY", "fake-key-for-test")
        content = '{"unsafe": true, "reason": "test reason"}'
        response = _FakeResponse(200, json_data={"choices": [{"message": {"content": content}}]})
        fake = _FakeAsyncClient(response=response)
        monkeypatch.setattr(guardrails_module.httpx, "AsyncClient", lambda **kw: fake)

        assert await _moderate_content("some answer text") == "test reason"

    @pytest.mark.asyncio
    async def test_safe_response_returns_none(self, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_API_KEY", "fake-key-for-test")
        content = '{"unsafe": false, "reason": ""}'
        response = _FakeResponse(200, json_data={"choices": [{"message": {"content": content}}]})
        fake = _FakeAsyncClient(response=response)
        monkeypatch.setattr(guardrails_module.httpx, "AsyncClient", lambda **kw: fake)

        assert await _moderate_content("some answer text") is None
