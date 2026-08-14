"""
Smoke Test Script

Tests basic functionality of all components.
"""

import asyncio
import logging
from datetime import datetime

from pipeline import process_query, QueryRequest
from config import configure_logging, settings

logger = logging.getLogger(__name__)


async def main():
    """Run smoke tests on all components."""
    start_time = datetime.utcnow()
    
    try:
        # Configure logging
        configure_logging("DEBUG")
        
        logger.info("Starting smoke tests...")
        
        # Test 1: Query processing pipeline
        logger.info("Test 1: Query processing pipeline")
        test_query = QueryRequest(
            query="Who discovered penicillin?",
            language="en",
        )
        
        try:
            response = await process_query(test_query)
            logger.info(f"Query test passed: {response.query[:50]}...")
            logger.info(f"Answer: {response.answer}")
            logger.info(f"Latency: {response.latency_ms:.2f}ms")
        except Exception as e:
            logger.error(f"Query test failed: {e}")
            return 1
        
        end_time = datetime.utcnow()
        duration = end_time - start_time
        
        logger.info(f"Smoke tests completed in {duration.total_seconds():.2f} seconds")
        return 0
        
    except Exception as e:
        logger.error(f"Smoke tests failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
