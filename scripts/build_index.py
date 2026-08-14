"""
Index Building Script

Downloads dataset, processes it, and builds indexes.
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime

from ingestion import download_dataset, normalize_data, chunk_documents, generate_embeddings, build_index
from config import configure_logging, settings

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for index building."""
    start_time = datetime.utcnow()
    
    try:
        # Configure logging
        configure_logging()
        
        logger.info("Starting index building process...")
        logger.info(f"Settings: APP_ENV={settings.APP_ENV}, LOG_LEVEL={settings.LOG_LEVEL}")
        
        # Step 1: Download dataset
        logger.info("Step 1/4: Downloading dataset...")
        records = await download_dataset(
            output_dir=Path(settings.DATASET_CACHE_DIR),
            dataset_name=settings.DATASET_NAME,
            split=settings.DATASET_SPLIT,
            limit=settings.DATASET_LIMIT,
            num_language_shards=settings.DATASET_NUM_SHARDS,
            shard_download_timeout_s=settings.DATASET_DOWNLOAD_TIMEOUT_S,
        )
        logger.info(f"Downloaded {len(records)} records")

        # Step 2: Normalize data
        logger.info("Step 2/5: Normalizing data...")
        normalized_records = normalize_data(records, preserve_metadata=True)
        logger.info(f"Normalized {len(normalized_records)} records")

        # Step 3: Chunk data (hybrid: sentence + token-window + metadata-adaptive)
        logger.info("Step 3/5: Chunking data...")
        chunked_records = chunk_documents(normalized_records)
        logger.info(f"Chunked into {len(chunked_records)} records")

        # Step 4: Generate embeddings
        logger.info("Step 4/5: Generating embeddings...")
        embedded_records = await generate_embeddings(documents=chunked_records)
        logger.info(f"Generated embeddings for {len(embedded_records)} records")

        # Step 5: Build indexes
        logger.info("Step 5/5: Building indexes...")
        qdrant_store, bm25_store = await build_index(documents=embedded_records)
        
        end_time = datetime.utcnow()
        duration = end_time - start_time
        
        logger.info(f"Index building completed successfully!")
        logger.info(f"Total time: {duration.total_seconds():.2f} seconds")
        logger.info(f"Records processed: {len(embedded_records)}")
        logger.info(f"Qdrant collection: {settings.QDRANT_COLLECTION}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Index building failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
