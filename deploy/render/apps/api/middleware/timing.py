"""
Timing Middleware

Measures request latency and adds timing headers.
"""

import time
from fastapi import Request
import logging

logger = logging.getLogger(__name__)


async def timing_middleware(request: Request, call_next):
    """
    Middleware to measure request processing time.
    
    Args:
        request: Incoming FastAPI request
        call_next: Next middleware/endpoint in chain
        
    Returns:
        Response with timing headers
    """
    start_time = time.perf_counter()
    
    response = await call_next(request)
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    # Add timing header
    response.headers["X-Response-Time"] = f"{latency_ms:.2f}ms"
    
    # Log timing for debugging
    logger.debug(f"Request {request.method} {request.url.path} took {latency_ms:.2f}ms")
    
    return response
