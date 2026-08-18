"""
Embedding Engine

Single source of truth for turning text into vectors. Both the offline
ingestion pipeline (batch-encoding documents) and the online query path
(encoding one user question) call into this module, so index-time and
query-time vectors are guaranteed to come from the identical model, device,
and normalization settings. Previously these lived as two separate mock
hash functions in ingestion/embed.py and retrieval/qdrant_store.py — subtly
different implementations here would silently break retrieval, since
dense search only works if both sides embed the same way.
"""

import asyncio
import logging
import threading
from typing import List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when the embedding model fails to load or encode text."""
    pass


_model = None
_model_lock = threading.Lock()


def _resolve_device(preference: str) -> str:
    if preference != "auto":
        return preference
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_model():
    """Lazy singleton load, guarded so concurrent first-requests don't each load a copy."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:  # re-check: another thread may have loaded it while we waited
            return _model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "sentence-transformers is not installed. Run "
                "`pip install sentence-transformers` (it's in requirements.txt)."
            ) from e

        model_name = settings.EMBEDDING_MODEL
        device = _resolve_device(settings.EMBEDDING_DEVICE)
        logger.info(
            f"Loading embedding model '{model_name}' on device '{device}' "
            "(first run downloads model weights, can take a minute)..."
        )
        try:
            model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model '{model_name}': {e}") from e

        model.max_seq_length = settings.EMBEDDING_MAX_SEQ_LENGTH
        actual_dim = model.get_embedding_dimension()
        if actual_dim != settings.EMBEDDING_DIMENSION:
            logger.warning(
                f"Configured EMBEDDING_DIMENSION={settings.EMBEDDING_DIMENSION} does not "
                f"match model '{model_name}''s real dimension={actual_dim}. The real "
                f"dimension is used everywhere; update .env to silence this warning."
            )

        _model = model
        logger.info(f"Embedding model ready (dim={actual_dim}, device={device})")
        return _model


def get_embedding_dimension() -> int:
    """The real output dimension of the configured model (may differ from settings.EMBEDDING_DIMENSION)."""
    if settings.EMBEDDING_BACKEND == "onnx":
        from retrieval import onnx_embedder
        return onnx_embedder.get_dimension(settings.EMBEDDING_ONNX_DIR, settings.EMBEDDING_MAX_SEQ_LENGTH)
    return _load_model().get_embedding_dimension()


def _encode_sync(texts: List[str], batch_size: int) -> List[List[float]]:
    # A true empty string embeds fine for most models but some sentence-transformers
    # backends warn/behave oddly on it; a single space is a harmless stand-in that
    # keeps the output list aligned 1:1 with the input list.
    cleaned = [t if isinstance(t, str) and t.strip() else " " for t in texts]

    if settings.EMBEDDING_BACKEND == "onnx":
        # Verified numerically equivalent to the torch path (cosine_sim=1.0,
        # max_abs_diff=0.0 across English/Hindi/empty/long-truncated cases —
        # see scripts/verify_onnx_embedder.py) — same weights, lighter runtime.
        from retrieval import onnx_embedder
        try:
            return onnx_embedder.encode(
                cleaned,
                model_dir=settings.EMBEDDING_ONNX_DIR,
                max_seq_length=settings.EMBEDDING_MAX_SEQ_LENGTH,
                batch_size=batch_size,
            )
        except onnx_embedder.OnnxEmbedderError as e:
            raise EmbeddingError(str(e)) from e

    model = _load_model()
    vectors = model.encode(
        cleaned,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


async def embed_texts(texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
    """
    Embed a list of texts into normalized vectors, preserving input order.

    Runs the blocking (CPU/GPU-bound) model call in a worker thread so it
    never blocks the FastAPI event loop while other requests are in flight.
    """
    if not texts:
        return []

    effective_batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

    try:
        return await asyncio.to_thread(_encode_sync, texts, effective_batch_size)
    except EmbeddingError:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        raise EmbeddingError(f"Embedding generation failed: {e}") from e


class _EmbeddingBatcher:
    """
    Coalesces concurrent embed_query() calls arriving within a short
    window into one batched model.encode() call (roadmap Phase 22). Every
    call still goes through the identical _encode_sync() that
    embed_texts() uses — batching changes only how inputs are grouped
    before that call, never the resulting vectors (see
    tests/test_embeddings.py for the equivalence proof).

    A batch flushes on whichever comes first: max_batch_size callers have
    joined it, or window_s has elapsed since the first caller joined.
    """

    def __init__(self, window_s: float, max_batch_size: int):
        self._window_s = window_s
        self._max_batch_size = max_batch_size
        self._pending: List[Tuple[str, "asyncio.Future[List[float]]"]] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    async def embed(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        future: "asyncio.Future[List[float]]" = loop.create_future()

        async with self._lock:
            self._pending.append((text, future))
            if len(self._pending) >= self._max_batch_size:
                self._trigger_immediate_flush_locked()
            elif self._flush_task is None:
                self._flush_task = asyncio.create_task(self._flush_after_delay())

        return await future

    def _trigger_immediate_flush_locked(self) -> None:
        """Caller must already hold self._lock."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None
        asyncio.create_task(self._flush())

    async def _flush_after_delay(self) -> None:
        try:
            await asyncio.sleep(self._window_s)
        except asyncio.CancelledError:
            return  # an immediate (max-batch-size) flush already claimed this batch
        await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            batch = self._pending
            self._pending = []
            self._flush_task = None
        if not batch:
            return

        texts = [t for t, _ in batch]
        try:
            vectors = await asyncio.to_thread(_encode_sync, texts, len(texts))
        except Exception as e:
            error = e if isinstance(e, EmbeddingError) else EmbeddingError(f"Embedding generation failed: {e}")
            for _, future in batch:
                if not future.done():
                    future.set_exception(error)
            return

        for (_, future), vector in zip(batch, vectors):
            if not future.done():
                future.set_result(vector)


_batcher: Optional[_EmbeddingBatcher] = None
_batcher_lock = threading.Lock()


def _get_batcher() -> _EmbeddingBatcher:
    """Lazy singleton, guarded by threading.Lock — same reasoning as _load_model()/_model_lock above."""
    global _batcher
    if _batcher is None:
        with _batcher_lock:
            if _batcher is None:
                _batcher = _EmbeddingBatcher(
                    window_s=settings.EMBEDDING_MICROBATCH_WINDOW_MS / 1000,
                    max_batch_size=settings.EMBEDDING_MICROBATCH_MAX_SIZE,
                )
    return _batcher


async def embed_query(query: str) -> List[float]:
    """Embed a single query string — the hot path used on every search request."""
    if not query or not query.strip():
        raise EmbeddingError("Cannot embed an empty query")
    if settings.EMBEDDING_MICROBATCH_ENABLED:
        return await _get_batcher().embed(query)
    vectors = await embed_texts([query], batch_size=1)
    return vectors[0]


def preload() -> None:
    """Force the model to load now instead of on the first request. Call at app startup."""
    if settings.EMBEDDING_BACKEND == "onnx":
        from retrieval import onnx_embedder
        onnx_embedder.preload(settings.EMBEDDING_ONNX_DIR, settings.EMBEDDING_MAX_SEQ_LENGTH)
        return
    _load_model()
