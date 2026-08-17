"""
STT Tests

Covers the pure/offline-testable pieces of pipeline/stt.py. A real
transcription call needs live audio + Groq credentials and is exercised
manually (see tests/test_generation.py's identical convention).
"""

import asyncio

import pytest
import pipeline.stt as stt_module
from config import settings


class TestSttSemaphore:
    """Roadmap Phase 22: bounds concurrent in-flight Groq Whisper calls, independently of the LLM semaphore."""

    @pytest.mark.asyncio
    async def test_get_stt_semaphore_is_a_singleton(self, monkeypatch):
        monkeypatch.setattr(stt_module, "_stt_semaphore", None)
        sem1 = stt_module._get_stt_semaphore()
        sem2 = stt_module._get_stt_semaphore()
        assert sem1 is sem2

    @pytest.mark.asyncio
    async def test_semaphore_bounds_concurrency_to_configured_limit(self, monkeypatch):
        monkeypatch.setattr(stt_module, "_stt_semaphore", None)
        monkeypatch.setattr(settings, "GROQ_STT_MAX_CONCURRENT", 2)

        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        async def task():
            nonlocal in_flight, max_in_flight
            async with stt_module._get_stt_semaphore():
                async with lock:
                    in_flight += 1
                    max_in_flight = max(max_in_flight, in_flight)
                await asyncio.sleep(0.03)
                async with lock:
                    in_flight -= 1

        await asyncio.gather(*(task() for _ in range(6)))
        assert max_in_flight == 2
