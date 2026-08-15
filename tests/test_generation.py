"""
Generation Tests

Tests for the generation module's pure/offline-testable pieces: context
building and JSON response parsing/validation. Tests that would need a real
Groq call are exercised manually (see IMPLEMENTATION_ROADMAP.md).
"""

import pytest
from pipeline.generation import build_context, _parse_and_validate, _MalformedResponse
from pipeline.schemas import Evidence


def test_build_context_includes_document_ids():
    evidence = [
        Evidence(passage="Evidence 1", score=0.9, source="qdrant", document_id="abc", language="en"),
        Evidence(passage="Evidence 2", score=0.8, source="bm25", document_id="def", language="en"),
    ]
    context = build_context(evidence)
    assert "[abc] Evidence 1" in context
    assert "[def] Evidence 2" in context


def test_build_context_empty():
    assert build_context([]) == ""


class TestParseAndValidate:
    def test_valid_json_parses_cleanly(self):
        content = '{"answer": "Fleming discovered it.", "evidence_ids": ["1"], "grounded": true, "confidence": 0.9}'
        result = _parse_and_validate(content, valid_evidence_ids=["1", "2"])
        assert result["answer"] == "Fleming discovered it."
        assert result["evidence_ids"] == ["1"]
        assert result["grounded"] is True
        assert result["confidence"] == 0.9

    def test_malformed_json_raises(self):
        with pytest.raises(_MalformedResponse):
            _parse_and_validate("not json at all", valid_evidence_ids=["1"])

    def test_missing_answer_raises(self):
        with pytest.raises(_MalformedResponse):
            _parse_and_validate('{"evidence_ids": [], "grounded": false, "confidence": 0}', valid_evidence_ids=[])

    def test_empty_answer_raises(self):
        with pytest.raises(_MalformedResponse):
            _parse_and_validate('{"answer": "   "}', valid_evidence_ids=[])

    def test_hallucinated_evidence_ids_are_dropped(self):
        """A cited id that was never actually retrieved must never be trusted."""
        content = '{"answer": "x", "evidence_ids": ["real", "made_up"], "grounded": true, "confidence": 0.9}'
        result = _parse_and_validate(content, valid_evidence_ids=["real"])
        assert result["evidence_ids"] == ["real"]

    def test_grounded_forced_false_if_no_real_evidence_cited(self):
        """The model claiming grounded=true while citing nothing real must not be trusted."""
        content = '{"answer": "x", "evidence_ids": ["made_up"], "grounded": true, "confidence": 0.9}'
        result = _parse_and_validate(content, valid_evidence_ids=["real"])
        assert result["grounded"] is False

    def test_confidence_out_of_range_is_clamped(self):
        content = '{"answer": "x", "evidence_ids": [], "grounded": false, "confidence": 5.0}'
        result = _parse_and_validate(content, valid_evidence_ids=[])
        assert result["confidence"] == 1.0

    def test_non_numeric_confidence_defaults_to_zero(self):
        content = '{"answer": "x", "evidence_ids": [], "grounded": false, "confidence": "high"}'
        result = _parse_and_validate(content, valid_evidence_ids=[])
        assert result["confidence"] == 0.0

    def test_non_list_evidence_ids_raises(self):
        with pytest.raises(_MalformedResponse):
            _parse_and_validate('{"answer": "x", "evidence_ids": "not a list"}', valid_evidence_ids=[])

    def test_prompt_injection_in_evidence_does_not_grant_extra_trust(self):
        """
        Simulates a retrieved passage containing an injection attempt. This
        module only validates the model's OUTPUT shape — the actual
        injection defense lives in the system prompt — but the output
        validator must still reject ids that weren't really retrieved
        regardless of what the "evidence" claimed.
        """
        content = (
            '{"answer": "ignored", "evidence_ids": ["real", "system_prompt_override"], '
            '"grounded": true, "confidence": 1.0}'
        )
        result = _parse_and_validate(content, valid_evidence_ids=["real"])
        assert "system_prompt_override" not in result["evidence_ids"]
