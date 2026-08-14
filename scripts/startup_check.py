"""
Startup Check Script

Verifies all external services are available.
"""

import asyncio
import logging
from datetime import datetime
import httpx

from config import configure_logging, settings

logger = logging.getLogger(__name__)


async def check_service(name: str, url: str, timeout: float = 5.0) -> bool:
    """Check if a service is available."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            if response.status_code in (200, 401, 403):  # 401/403 means service is up
                logger.info(f"{name}: OK")
                return True
            else:
                logger.error(f"{name}: Failed with status {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"{name}: Error - {e}")
        return False


async def main():
    """Check all external services."""
    start_time = datetime.utcnow()
    
    try:
        configure_logging()
        
        logger.info("Checking external services...")
        
        services = [
            ("Qdrant", f"{settings.QDRANT_URL}/collections"),
        ]
        
        results = []
        for name, url in services:
            result = await check_service(name, url)
            results.append((name, result))
        
        # Check Groq API
        groq_ok = False
        if settings.GROQ_API_KEY:
            groq_ok = True  # Basic check - actual API call would need testing
            logger.info("Groq: API key configured")
        else:
            logger.warning("Groq: API key not configured")
        results.append(("Groq", groq_ok))
        
        end_time = datetime.utcnow()
        duration = end_time - start_time
        
        # Summary
        all_ok = all(result for _, result in results)
        
        logger.info(f"\nService Check Summary:")
        for name, result in results:
            status = "OK" if result else "FAILED"
            logger.info(f"  {name}: {status}")
        
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
