"""
Configuration Module

Handles application configuration and settings.
"""

from .settings import settings
from .logging import configure_logging

__all__ = ["settings", "configure_logging"]
