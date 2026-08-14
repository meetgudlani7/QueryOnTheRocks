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
from typing import List, Optional

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
    return _load_model().get_embedding_dimension()


def _encode_sync(texts: List[str], batch_size: int) -> List[List[float]]:
    model = _load_model()
    # A true empty string embeds fine for most models but some sentence-transformers
    # backends warn/behave oddly on it; a single space is a harmless stand-in that
    # keeps the output list aligned 1:1 with the input list.
    cleaned = [t if isinstance(t, str) and t.strip() else " " for t in texts]
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


async def embed_query(query: str) -> List[float]:
    """Embed a single query string — the hot path used on every search request."""
    if not query or not query.strip():
        raise EmbeddingError("Cannot embed an empty query")
    vectors = await embed_texts([query], batch_size=1)
    return vectors[0]


def preload() -> None:
    """Force the model to load now instead of on the first request. Call at app startup."""
    _load_model()
