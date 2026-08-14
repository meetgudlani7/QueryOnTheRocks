"""
API Routes Package

Contains all route definitions for the FastAPI backend.
"""

from .health import router as health_router
from .audio import router as audio_router
from .query import router as query_router

__all__ = ["health_router", "audio_router", "query_router"]
