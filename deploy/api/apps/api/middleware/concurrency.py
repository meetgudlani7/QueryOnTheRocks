"""
Concurrency Ceiling Middleware

Bounds concurrent in-flight requests to /api/* routes (roadmap Phase 22)
— protects the shared, process-wide embedding model and Qdrant/BM25
connections from unbounded pile-up under a traffic burst. A request that
can't be admitted within CONCURRENCY_QUEUE_TIMEOUT_MS gets a clean 503
instead of silently degrading every other in-flight request's latency.

Scoped to /api/* only — /health, /health/ready, /, and /docs must stay
responsive even under heavy query load, since those are exactly what a
monitoring/readiness check depends on to notice something's wrong.

Registration order matters here (see apps/api/main.py): this middleware
must be registered *before* CORSMiddleware so CORS ends up wrapping it —
Starlette's last-registered middleware becomes the outermost layer, so
registering this first means CORS (registered after) still gets to add
Access-Control-* headers to a 503 this middleware generates. Registered
after this one but wrapped by CORS either way, timing_middleware still
correctly measures total latency including any time spent waiting here.
"""

import asyncio
import logging
import threading
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from config import settings

logger = logging.getLogger(__name__)

_semaphore: Optional[asyncio.Semaphore] = None
_semaphore_lock = threading.Lock()


def _get_semaphore() -> asyncio.Semaphore:
    """Lazy singleton, guarded by threading.Lock — same reasoning as retrieval/embeddings.py's _load_model()/_model_lock."""
    global _semaphore
    if _semaphore is None:
        with _semaphore_lock:
            if _semaphore is None:
                _semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
    return _semaphore


async def concurrency_limit_middleware(request: Request, call_next):
    """
    Admits a request only if a concurrency slot is free within
    CONCURRENCY_QUEUE_TIMEOUT_MS; otherwise returns 503 immediately
    rather than queueing indefinitely and letting every other in-flight
    request's latency degrade unboundedly under a sustained burst.

    MAX_CONCURRENT_REQUESTS <= 0 disables the ceiling entirely (today's
    unbounded behavior, unchanged).
    """
    if settings.MAX_CONCURRENT_REQUESTS <= 0 or not request.url.path.startswith("/api/"):
        return await call_next(request)

    semaphore = _get_semaphore()
    timeout_s = settings.CONCURRENCY_QUEUE_TIMEOUT_MS / 1000

    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning(
            f"Request to {request.url.path} rejected: at capacity "
            f"({settings.MAX_CONCURRENT_REQUESTS} concurrent requests)"
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "Server is at capacity, please retry shortly."},
        )

    try:
        return await call_next(request)
    finally:
        semaphore.release()
