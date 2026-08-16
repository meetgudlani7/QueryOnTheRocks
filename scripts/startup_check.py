"""
Startup Check Script

Verifies all external services are available.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
import httpx

# Running this file directly (`python scripts/startup_check.py`) puts this
# file's own directory on sys.path, not the project root — without this,
# `from config import ...` below fails with ModuleNotFoundError. Mirrors
# scripts/build_index.py's identical bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import configure_logging, settings
from retrieval import qdrant_store

logger = logging.getLogger(__name__)


async def check_qdrant() -> bool:
    """Live probe against the configured collection, not just the Qdrant host."""
    try:
        ok = await qdrant_store.ping()
        logger.info(f"Qdrant: {'OK' if ok else 'FAILED'} (collection '{settings.QDRANT_COLLECTION}')")
        return ok
    except Exception as e:
        logger.error(f"Qdrant: Error - {e}")
        return False


async def check_groq() -> bool:
    """Live models-list call, not just a check that a key string is present."""
    if not settings.GROQ_API_KEY:
        logger.warning("Groq: API key not configured")
        return False
    try:
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_MS / 1000) as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
            )
        ok = response.status_code == 200
        logger.info(f"Groq: {'OK' if ok else f'FAILED ({response.status_code})'}")
        return ok
    except httpx.HTTPError as e:
        logger.error(f"Groq: Error - {e}")
        return False


async def main():
    """Check all external services."""
    start_time = datetime.now(timezone.utc)

    try:
        configure_logging()
        logger.info("Checking external services...")

        results = [
            ("Qdrant", await check_qdrant()),
            ("Groq", await check_groq()),
        ]

        duration = datetime.now(timezone.utc) - start_time
        all_ok = all(result for _, result in results)

        logger.info("\nService Check Summary:")
        for name, result in results:
            logger.info(f"  {name}: {'OK' if result else 'FAILED'}")

        logger.info(f"\nChecks completed in {duration.total_seconds():.2f} seconds")

        if all_ok:
            logger.info("All services are healthy!")
            return 0
        else:
            logger.error("Some services are not available")
            return 1

    except Exception as e:
        logger.error(f"Startup check failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
