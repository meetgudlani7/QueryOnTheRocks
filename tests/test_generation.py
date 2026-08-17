"""
Generation Tests

Tests for the generation module's pure/offline-testable pieces: context
building and JSON response parsing/validation. Tests that would need a real
Groq call are exercised manually (see IMPLEMENTATION_ROADMAP.md).
"""

import asyncio

import pytest
from pipeline.generation import (
    build_context,
    _parse_and_validate,
    _MalformedResponse,
    _split_stream_output,
    _stream_token_step,
    STREAM_DELIMITER,
)
from pipeline.schemas import Evidence
import pipeline.generation as generation_module
from config import settings


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


class TestSplitStreamOutput:
    def test_splits_on_delimiter(self):
        full = f'Paris, France.\n{STREAM_DELIMITER}\n{{"evidence_ids": ["e1"], "grounded": true, "confidence": 0.9}}'
        answer, meta = _split_stream_output(full)
        assert answer == "Paris, France."
        assert meta == '{"evidence_ids": ["e1"], "grounded": true, "confidence": 0.9}'

    def test_missing_delimiter_returns_whole_text_as_answer(self):
        answer, meta = _split_stream_output("just plain text, no delimiter")
        assert answer == "just plain text, no delimiter"
        assert meta == ""


class TestStreamTokenStep:
    """
    Regression coverage for a real bug found in live testing: Groq's SSE
    deltas can split STREAM_DELIMITER across multiple chunks (e.g.
    "<<<META" then ">>>" as two separate deltas). An earlier version only
    checked whether the *complete* delimiter was present in the
    accumulated text before deciding what to emit, which let the first
    delta's tail ("<<<META") leak into the visible token stream before the
    second delta ever arrived. _stream_token_step must withhold enough of
    a trailing lookback window to never emit a partial delimiter match.
    """

    def test_emits_plain_text_immediately_once_past_the_lookback_window(self):
        """
        Text shorter than or equal to the lookback window (len(delimiter)-1
        characters) is conservatively withheld entirely, since it alone
        can't yet be ruled out as the start of a delimiter — only once
        enough text has accumulated does the safely-confirmed prefix start
        emitting.
        """
        long_text = "Hello, this plain text is much longer than the lookback window."
        text, emitted_len, seen = _stream_token_step(long_text, 0, False)
        assert text is not None
        assert seen is False
        # Exactly the lookback window's worth of trailing characters is
        # withheld; everything before that is safe to emit immediately.
        lookback = len(STREAM_DELIMITER) - 1
        assert text == long_text[: len(long_text) - lookback]
        assert emitted_len == len(long_text) - lookback

    def test_never_leaks_a_delimiter_split_across_two_deltas(self):
        """The exact scenario observed live: delimiter split mid-string."""
        split_point = len(STREAM_DELIMITER) - 1  # leaves exactly 1 char of the delimiter for the next delta
        answer_prefix = "Paris, France.\n"
        first_delta_text = answer_prefix + STREAM_DELIMITER[:split_point]
        second_delta_text = first_delta_text + STREAM_DELIMITER[split_point:] + '\n{"grounded": true}'

        text1, emitted_len, seen = _stream_token_step(first_delta_text, 0, False)
        assert text1 is not None
        assert STREAM_DELIMITER not in text1
        # None of the delimiter's characters may appear at all in what's
        # emitted from this first, still-ambiguous chunk.
        assert not any(text1.endswith(STREAM_DELIMITER[:i]) and i > 0 for i in range(1, len(STREAM_DELIMITER)))
        assert seen is False

        text2, emitted_len, seen = _stream_token_step(second_delta_text, emitted_len, seen)
        assert seen is True
        combined = (text1 or "") + (text2 or "")
        # The trailing newline before the delimiter is real pre-delimiter
        # content and is expected in the emitted stream, exactly matching
        # observed live behavior — only STREAM_DELIMITER itself is
        # withheld, not surrounding whitespace.
        assert combined == answer_prefix
        assert STREAM_DELIMITER not in combined

    def test_full_delimiter_in_one_delta_stops_emission_at_boundary(self):
        text, emitted_len, seen = _stream_token_step(f"answer text{STREAM_DELIMITER}meta", 0, False)
        assert text == "answer text"
        assert seen is True

    def test_no_further_emission_once_delimiter_seen(self):
        text, emitted_len, seen = _stream_token_step("anything", 5, True)
        assert text is None
        assert seen is True

    def test_incremental_feed_of_full_answer_never_leaks_delimiter_fragment(self):
        """
        Simulates feeding the delimiter one character at a time (the most
        adversarial possible chunking) and asserts the concatenation of
        everything ever emitted, plus whatever's still held back at the
        end, exactly reconstructs the pre-delimiter answer with zero
        delimiter characters ever appearing in an emitted token.
        """
        full = "The answer is here." + STREAM_DELIMITER + '{"grounded": true}'
        accumulated = ""
        emitted_len = 0
        seen = False
        emitted_chunks = []
        for ch in full:
            accumulated += ch
            text, emitted_len, seen = _stream_token_step(accumulated, emitted_len, seen)
            if text:
                emitted_chunks.append(text)
            if seen:
                break

        combined = "".join(emitted_chunks)
        assert combined == "The answer is here."
        assert STREAM_DELIMITER not in combined
        for i in range(1, len(STREAM_DELIMITER)):
            assert not combined.endswith(STREAM_DELIMITER[:i])


class TestLlmSemaphore:
    """
    Roadmap Phase 22: bounds concurrent in-flight Groq calls. Tested here
    as pure asyncio coordination — no real Groq calls, no httpx mocking
    (matching this module's established "network calls verified manually,
    pure logic gets unit tests" convention) — just proving the semaphore
    singleton and its concurrency bound behave correctly in isolation.
    """

    @pytest.mark.asyncio
    async def test_get_llm_semaphore_is_a_singleton(self, monkeypatch):
        monkeypatch.setattr(generation_module, "_llm_semaphore", None)
        sem1 = generation_module._get_llm_semaphore()
        sem2 = generation_module._get_llm_semaphore()
        assert sem1 is sem2

    @pytest.mark.asyncio
    async def test_semaphore_bounds_concurrency_to_configured_limit(self, monkeypatch):
        monkeypatch.setattr(generation_module, "_llm_semaphore", None)
        monkeypatch.setattr(settings, "GROQ_LLM_MAX_CONCURRENT", 2)

        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        async def task():
            nonlocal in_flight, max_in_flight
            async with generation_module._get_llm_semaphore():
                async with lock:
                    in_flight += 1
                    max_in_flight = max(max_in_flight, in_flight)
                await asyncio.sleep(0.03)
                async with lock:
                    in_flight -= 1

        await asyncio.gather(*(task() for _ in range(6)))
        assert max_in_flight == 2

    @pytest.mark.asyncio
    async def test_semaphore_releases_after_use(self, monkeypatch):
        """No leaked slots — full capacity must be available again afterward."""
        monkeypatch.setattr(generation_module, "_llm_semaphore", None)
        monkeypatch.setattr(settings, "GROQ_LLM_MAX_CONCURRENT", 3)
        sem = generation_module._get_llm_semaphore()

        async def task():
            async with sem:
                await asyncio.sleep(0.01)

        await asyncio.gather(*(task() for _ in range(9)))
        assert sem._value == 3
