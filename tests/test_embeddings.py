"""
Embedding Engine Tests

Covers the pure coordination logic of the Phase 22 micro-batcher
(_EmbeddingBatcher) with a fake encode function — no real model load,
matching this project's established "model/network calls verified
manually, coordination logic gets fast automated tests" convention (see
tests/test_generation.py's identical approach for Groq streaming).

Real equivalence between batched and unbatched output — proving batching
never changes the actual vectors, only how inputs are grouped before the
identical underlying model call — was verified live against the real
model; see OPTIMIZATION_ROADMAP.md Phase 22 for that result.
"""

import asyncio

import pytest
import retrieval.embeddings as emb
from retrieval.embeddings import _EmbeddingBatcher, EmbeddingError


class TestEmbeddingBatcher:
    @pytest.mark.asyncio
    async def test_concurrent_calls_coalesce_into_one_underlying_call(self, monkeypatch):
        call_log = []

        def fake_encode_sync(texts, batch_size):
            call_log.append(list(texts))
            return [[float(len(t))] for t in texts]

        monkeypatch.setattr(emb, "_encode_sync", fake_encode_sync)

        batcher = _EmbeddingBatcher(window_s=0.05, max_batch_size=32)
        await asyncio.gather(*(batcher.embed(f"text{i}") for i in range(5)))

        assert len(call_log) == 1  # exactly one underlying model call for all 5 concurrent callers
        assert call_log[0] == [f"text{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_each_caller_gets_its_own_correct_vector(self, monkeypatch):
        """Correct 1:1 mapping — no vector must ever get attached to the wrong caller."""

        def fake_encode_sync(texts, batch_size):
            return [[float(len(t)), float(hash(t) % 1000)] for t in texts]

        monkeypatch.setattr(emb, "_encode_sync", fake_encode_sync)

        batcher = _EmbeddingBatcher(window_s=0.05, max_batch_size=32)
        texts = [f"distinct text number {i}" for i in range(8)]
        results = await asyncio.gather(*(batcher.embed(t) for t in texts))

        for text, vector in zip(texts, results):
            assert vector == [float(len(text)), float(hash(text) % 1000)]

    @pytest.mark.asyncio
    async def test_flushes_immediately_at_max_batch_size(self, monkeypatch):
        """A long window must not matter once max_batch_size is reached."""
        call_log = []

        def fake_encode_sync(texts, batch_size):
            call_log.append(len(texts))
            return [[0.0] for _ in texts]

        monkeypatch.setattr(emb, "_encode_sync", fake_encode_sync)

        # window_s is far longer than this test's own timeout — if the
        # implementation were waiting for the window instead of the
        # max-batch-size trigger, this would hang and time out.
        batcher = _EmbeddingBatcher(window_s=5.0, max_batch_size=3)
        await asyncio.wait_for(
            asyncio.gather(*(batcher.embed(f"t{i}") for i in range(3))), timeout=2.0
        )
        assert call_log == [3]

    @pytest.mark.asyncio
    async def test_separate_batches_after_a_flush(self, monkeypatch):
        call_log = []

        def fake_encode_sync(texts, batch_size):
            call_log.append(len(texts))
            return [[0.0] for _ in texts]

        monkeypatch.setattr(emb, "_encode_sync", fake_encode_sync)

        batcher = _EmbeddingBatcher(window_s=0.02, max_batch_size=32)
        await asyncio.gather(*(batcher.embed(f"a{i}") for i in range(3)))
        await asyncio.gather(*(batcher.embed(f"b{i}") for i in range(2)))

        assert call_log == [3, 2]

    @pytest.mark.asyncio
    async def test_error_propagates_to_every_waiting_caller(self, monkeypatch):
        def failing_encode_sync(texts, batch_size):
            raise RuntimeError("model exploded")

        monkeypatch.setattr(emb, "_encode_sync", failing_encode_sync)

        batcher = _EmbeddingBatcher(window_s=0.02, max_batch_size=32)
        results = await asyncio.gather(
            *(batcher.embed(f"t{i}") for i in range(4)), return_exceptions=True
        )
        assert len(results) == 4
        assert all(isinstance(r, EmbeddingError) for r in results)

    @pytest.mark.asyncio
    async def test_batcher_is_reusable_after_a_flush(self, monkeypatch):
        """No leaked state — sequential use after a flush must start a clean, fresh batch."""

        def fake_encode_sync(texts, batch_size):
            return [[1.0] for _ in texts]

        monkeypatch.setattr(emb, "_encode_sync", fake_encode_sync)

        batcher = _EmbeddingBatcher(window_s=0.02, max_batch_size=32)
        assert await batcher.embed("one") == [1.0]
        assert await batcher.embed("two") == [1.0]
        assert batcher._pending == []
        assert batcher._flush_task is None


class TestEmbedQueryRouting:
    @pytest.mark.asyncio
    async def test_embed_query_bypasses_batcher_when_disabled(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "EMBEDDING_MICROBATCH_ENABLED", False)

        called = {"batcher": False}

        class _FakeBatcher:
            async def embed(self, text):
                called["batcher"] = True
                return [0.0]

        monkeypatch.setattr(emb, "_get_batcher", lambda: _FakeBatcher())
        monkeypatch.setattr(emb, "embed_texts", lambda texts, batch_size=None: asyncio.sleep(0, result=[[1.0]]))

        result = await emb.embed_query("hello")
        assert called["batcher"] is False
        assert result == [1.0]

    @pytest.mark.asyncio
    async def test_embed_query_uses_batcher_when_enabled(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "EMBEDDING_MICROBATCH_ENABLED", True)

        called = {"batcher": False}

        class _FakeBatcher:
            async def embed(self, text):
                called["batcher"] = True
                return [2.0]

        monkeypatch.setattr(emb, "_get_batcher", lambda: _FakeBatcher())

        result = await emb.embed_query("hello")
        assert called["batcher"] is True
        assert result == [2.0]

    @pytest.mark.asyncio
    async def test_embed_query_still_rejects_empty_string_regardless_of_flag(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "EMBEDDING_MICROBATCH_ENABLED", True)
        with pytest.raises(EmbeddingError):
            await emb.embed_query("   ")
