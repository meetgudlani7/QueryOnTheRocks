"""
Reranking Engine

Cross-encoder reranking of already-retrieved candidates (roadmap Phase
20). RRF fusion (retrieval/fusion.py) combines Qdrant and BM25 by rank
position only — it has no model of whether a candidate actually answers
the query, just where each search engine happened to place it. A
cross-encoder scores each (query, passage) pair jointly, which is a
strictly more expensive but strictly more informed relevance signal.
Applying it only to fusion's own output (a few dozen candidates) rather
than the whole corpus keeps that cost bounded.

Mirrors retrieval/embeddings.py's lazy-singleton, device-auto-detecting,
to_thread-offloaded pattern deliberately, so this module behaves exactly
like the piece of the system it sits next to. Gated behind
settings.RERANKING_ENABLED (default off) — see pipeline/retrieval.py's
rerank_evidence() for the fail-open caller-side behavior.
"""

import asyncio
import logging
import threading
from typing import List, Optional

from config import settings

logger = logging.getLogger(__name__)


class RerankerError(Exception):
    """Raised when the reranker model fails to load or score."""
    pass


_model = None
_model_lock = threading.Lock()


def _resolve_device(preference: str) -> str:
    # Identical logic to retrieval/embeddings.py's _resolve_device.
    # Duplicated rather than imported so this optional, feature-flagged
    # module has no import-time coupling to the embedding module's
    # internals — a change to one can't accidentally break the other.
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
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RerankerError(
                "sentence-transformers is not installed. Run "
                "`pip install sentence-transformers` (it's in requirements.txt)."
            ) from e

        model_name = settings.RERANKER_MODEL
        device = _resolve_device(settings.EMBEDDING_DEVICE)
        logger.info(
            f"Loading reranker model '{model_name}' on device '{device}' "
            "(first run downloads model weights, can take a minute)..."
        )
        try:
            model = CrossEncoder(model_name, device=device, max_length=512)
        except Exception as e:
            raise RerankerError(f"Failed to load reranker model '{model_name}': {e}") from e

        _model = model
        logger.info("Reranker model ready")
        return _model


def _score_sync(query: str, passages: List[str]) -> List[float]:
    model = _load_model()
    # A blank passage is a defensive guard, not an expected input — mirrors
    # embeddings.py's identical treatment of empty text.
    pairs = [(query, p if isinstance(p, str) and p.strip() else " ") for p in passages]
    scores = model.predict(pairs)
    return [float(s) for s in scores]


async def rerank(query: str, passages: List[str]) -> List[float]:
    """
    Score each (query, passage) pair, preserving input order — higher
    means more relevant. Raises RerankerError on any failure; callers
    (see pipeline/retrieval.py::rerank_evidence) are expected to catch it
    and fail open to the pre-rerank order rather than fail the request.

    Runs the blocking model call in a worker thread so it never blocks the
    FastAPI event loop while other requests are in flight — same reason
    retrieval/embeddings.py and retrieval/bm25_store.py do the same.
    """
    if not passages:
        return []
    if not query or not query.strip():
        raise RerankerError("Cannot rerank against an empty query")

    try:
        return await asyncio.to_thread(_score_sync, query, passages)
    except RerankerError:
        raise
    except Exception as e:
        logger.error(f"Reranking failed: {e}", exc_info=True)
        raise RerankerError(f"Reranking failed: {e}") from e


def preload() -> None:
    """Force the model to load now instead of on the first request. Call at app startup when enabled."""
    _load_model()
