"""
Middleware Package

Contains middleware for the FastAPI backend.
"""

from .timing import timing_middleware

__all__ = ["timing_middleware"]
