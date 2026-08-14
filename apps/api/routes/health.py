"""
Health Check Endpoint

Provides health status and basic system information.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/health", tags=["health"])


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


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.
    
    Verifies that all external services are available.
    
    Returns:
        dict: Readiness status
    """
    # TODO: Add actual service checks (Groq, Qdrant, etc.)
    return {
        "status": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "groq": "pending",
            "qdrant": "pending",
        },
    }
