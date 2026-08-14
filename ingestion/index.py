"""
Index Building Module

Builds Qdrant and BM25 indexes from embedded, normalized documents.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from retrieval import BM25Store, QdrantStore

logger = logging.getLogger(__name__)


class IndexBuildError(Exception):
    """Custom exception for index building errors."""
    pass


async def build_index(
    documents: List[Dict[str, Any]],
    qdrant_store: Optional[QdrantStore] = None,
    bm25_store: Optional[BM25Store] = None,
) -> Tuple[QdrantStore, BM25Store]:
    """
    Build Qdrant and BM25 indexes from documents that already carry an
    "embedding" vector (from ingestion.embed.generate_embeddings).

    Raises:
        IndexBuildError: if there's nothing to index, or any document is
            missing its embedding (indicates a broken upstream step, not
            something to silently skip).
    """
    if not documents:
        raise IndexBuildError("No documents to index")

    missing_embedding = [d.get("id") for d in documents if not d.get("embedding")]
    if missing_embedding:
        raise IndexBuildError(
            f"{len(missing_embedding)} document(s) have no embedding "
            f"(e.g. id={missing_embedding[0]}); run generate_embeddings() before build_index()"
        )

    start_time = time.perf_counter()
    qdrant_store = qdrant_store or QdrantStore()
    bm25_store = bm25_store or BM25Store()

    vector_size = len(documents[0]["embedding"])
    await qdrant_store.ensure_collection(vector_size=vector_size)

    qdrant_docs = [
        {
            "id": doc["id"],
            "vector": doc["embedding"],
            "payload": {
                "passage": doc.get("passage", ""),
                "language": doc.get("language", "unknown"),
                "query_type": doc.get("query_type"),
                "is_selected": doc.get("is_selected"),
                "metadata": doc.get("metadata", {}),
            },
        }
        for doc in documents
    ]
    logger.info(f"Indexing {len(qdrant_docs)} documents to Qdrant collection '{qdrant_store.collection}'...")
    await qdrant_store.upsert(qdrant_docs, batch_size=settings.QDRANT_UPSERT_BATCH_SIZE)

    bm25_docs = [
        {
            "id": doc["id"],
            "passage": doc.get("passage", ""),
            "language": doc.get("language", "unknown"),
            "metadata": doc.get("metadata", {}),
        }
        for doc in documents
    ]
    logger.info(f"Indexing {len(bm25_docs)} documents to BM25...")
    await bm25_store.add_documents(bm25_docs)
    bm25_store.save(settings.BM25_INDEX_PATH)

    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"Index building completed in {latency_ms:.2f}ms")

    return qdrant_store, bm25_store
