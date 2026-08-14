"""
Embedding Generation Module

Attaches real embedding vectors to normalized documents, using the shared
embedding engine in retrieval.embeddings so index-time vectors are produced
by the exact same model as query-time vectors (see that module's docstring
for why this matters).
"""

import logging
from typing import Any, Dict, List, Optional

from retrieval.embeddings import EmbeddingError, embed_texts, get_embedding_dimension
from config import settings

logger = logging.getLogger(__name__)

__all__ = ["generate_embeddings", "EmbeddingError"]


async def generate_embeddings(
    documents: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Embed a list of normalized documents (each expected to have a non-empty
    "passage" field) and attach the vector under "embedding".

    Documents with missing/empty passage text are skipped and logged rather
    than silently embedded as near-meaningless placeholder vectors — a bad
    upstream record should not poison the index with junk that ranks
    unpredictably in search.
    """
    if not documents:
        return []

    if model_name and model_name != settings.EMBEDDING_MODEL:
        logger.warning(
            f"generate_embeddings(model_name='{model_name}') was ignored — the shared "
            f"embedding engine always uses config EMBEDDING_MODEL="
            f"'{settings.EMBEDDING_MODEL}' so index-time and query-time vectors match. "
            f"Change EMBEDDING_MODEL in .env instead."
        )

    valid_docs = []
    skipped = 0
    for doc in documents:
        passage = doc.get("passage", "")
        if isinstance(passage, str) and passage.strip():
            valid_docs.append(doc)
        else:
            skipped += 1

    if skipped:
        logger.warning(f"Skipping {skipped} document(s) with empty/missing passage text")

    if not valid_docs:
        logger.warning("No valid documents to embed after filtering")
        return []

    texts = [doc["passage"] for doc in valid_docs]
    effective_batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

    logger.info(
        f"Embedding {len(texts)} documents (batch_size={effective_batch_size}, "
        f"model={settings.EMBEDDING_MODEL})..."
    )

    vectors = await embed_texts(texts, batch_size=effective_batch_size)

    if len(vectors) != len(valid_docs):
        # Should be structurally impossible since embed_texts preserves order/count,
        # but this guards against ever silently misaligning a vector to the wrong
        # passage, which would be a very hard bug to notice later.
        raise EmbeddingError(
            f"Embedding count mismatch: got {len(vectors)} vectors for {len(valid_docs)} documents"
        )

    embedded_docs = []
    for doc, vector in zip(valid_docs, vectors):
        new_doc = dict(doc)
        new_doc["embedding"] = vector
        embedded_docs.append(new_doc)

    logger.info(f"Generated {len(embedded_docs)} embeddings (dim={get_embedding_dimension()})")
    return embedded_docs
