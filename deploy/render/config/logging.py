"""
Logging Configuration

Configures logging for the application.
"""

import contextvars
import logging
import sys
from typing import Optional
from pathlib import Path

from .settings import settings

# Request-scoped id for log correlation (Phase 9). A ContextVar rather than
# a plain module global because it's automatically isolated per asyncio
# Task — concurrent requests never bleed their ids into each other's logs —
# and because it lets every logger anywhere in the call stack (retrieval,
# generation, guardrails, ...) pick it up for free, without threading a
# request_id parameter through every function signature.
request_id_var: "contextvars.ContextVar[str]" = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(log_level: Optional[str] = None) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Override log level (default from settings)
    """
    level = getattr(logging, log_level or settings.LOG_LEVEL, logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | [%(request_id)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    request_id_filter = _RequestIdFilter()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_id_filter)
    root_logger.addHandler(console_handler)
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # In production, also log to file
    if settings.APP_ENV == "production":
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "app.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(request_id_filter)
        root_logger.addHandler(file_handler)
    
    logging.info(f"Logging configured with level: {level}")
