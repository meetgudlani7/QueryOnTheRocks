"""
Rate Limiting Middleware (roadmap Phase 23)

In-process token-bucket rate limiting per client — the request's API key
(apps/api/dependencies.py) if present, otherwise client IP. Deliberately
in-process rather than an external store (Redis etc.): same reasoning as
the Phase 22 concurrency ceiling — this scale doesn't need a distributed
limiter yet, and an in-process one has no new failure mode (a broker
being down) to reason about.

Scoped to /api/* only, same reasoning and same registration-order
requirement as the concurrency ceiling (apps/api/middleware/concurrency.py)
— must be registered *before* CORSMiddleware in apps/api/main.py so a 429
this middleware generates still gets proper CORS headers.

Known simplification: _buckets grows by one entry per distinct client
ever seen, with no eviction. Acceptable at this project's current scale
(matches the "no external queue yet" posture elsewhere in this phase);
revisit with a proper TTL/eviction policy before this sees real
production traffic from many distinct clients.
"""

import logging
import threading
import time
from typing import Dict, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse

from config import settings

logger = logging.getLogger(__name__)

_buckets: Dict[str, Tuple[float, float]] = {}  # client_id -> (tokens_remaining, last_refill_monotonic)
_buckets_lock = threading.Lock()


def _client_id(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def _allow_request(client_id: str, limit_per_minute: int) -> bool:
    """
    Classic token bucket: refills continuously at limit_per_minute/60
    tokens per second, capped at limit_per_minute tokens total. Consumes
    one token and returns True if a request is currently allowed.
    """
    now = time.monotonic()
    refill_rate = limit_per_minute / 60.0

    with _buckets_lock:
        tokens, last_refill = _buckets.get(client_id, (float(limit_per_minute), now))
        elapsed = max(0.0, now - last_refill)
        tokens = min(float(limit_per_minute), tokens + elapsed * refill_rate)

        if tokens < 1.0:
            _buckets[client_id] = (tokens, now)
            return False

        _buckets[client_id] = (tokens - 1.0, now)
        return True


async def rate_limit_middleware(request: Request, call_next):
    """RATE_LIMIT_REQUESTS_PER_MINUTE <= 0 disables rate limiting entirely."""
    limit = settings.RATE_LIMIT_REQUESTS_PER_MINUTE
    if limit <= 0 or not request.url.path.startswith("/api/"):
        return await call_next(request)

    client_id = _client_id(request)
    if not _allow_request(client_id, limit):
        logger.warning(f"Rate limit exceeded for {client_id} on {request.url.path}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded, please slow down."},
            headers={"Retry-After": "60"},
        )

    return await call_next(request)
