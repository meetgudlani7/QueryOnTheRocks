"""
Health Check Endpoint

Provides health status and readiness checks against live dependencies.
"""

import asyncio
import logging
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter

from config import settings
from retrieval import qdrant_store

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger(__name__)


@router.get("")
async def health_check():
    """
    Health check endpoint.

    Returns:
        dict: Health status with timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


async def _check_groq() -> str:
    """Lightweight probe: list models, just to confirm the API key and network path work."""
    if not settings.GROQ_API_KEY:
        return "degraded"
    try:
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_MS / 1000) as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
            )
        return "ok" if response.status_code == 200 else "degraded"
    except httpx.HTTPError as e:
        logger.warning(f"Groq health check failed: {e}")
        return "degraded"


async def _check_qdrant() -> str:
    try:
        return "ok" if await qdrant_store.ping() else "degraded"
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        return "degraded"


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.

    Pings Groq and Qdrant directly so a downed dependency is reflected here,
    not just discovered on the next real request.

    Returns:
        dict: Readiness status per service
    """
    groq_status, qdrant_status = await asyncio.gather(_check_groq(), _check_qdrant())
    services = {"groq": groq_status, "qdrant": qdrant_status}

    return {
        "status": "ok" if all(s == "ok" for s in services.values()) else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": services,
    }
