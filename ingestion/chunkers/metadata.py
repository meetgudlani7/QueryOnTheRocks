"""
Metadata-Aware Chunker (Strategy C)

Unlike Strategy A/B, this one lets dataset metadata drive the chunking
*decision*, not just get attached afterward:

- A passage marked `is_selected=True` is MS MARCO's gold evidence passage
  for its query — the one that actually contains the answer. Splitting it
  risks separating the answer from the context that supports it, so these
  are kept as a single, whole, unsplit chunk.
- Everything else (distractor passages) is safe to split for finer-grained
  retrieval, so it's delegated to the token-window (semantic) chunker.

Every chunk carries full metadata: chunk_id, language, query_id,
query_type, is_selected, chunk_strategy.
"""

import logging
from typing import Any, Dict, List, Optional

from .semantic import SemanticChunker

logger = logging.getLogger(__name__)


class MetadataChunker:
    """Metadata-driven chunking: protects gold passages, delegates the rest."""

    def __init__(
        self,
        semantic_chunker: Optional[SemanticChunker] = None,
        language: str = "en",
    ):
        self.semantic_chunker = semantic_chunker or SemanticChunker()
        self.language = language

    def chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Args:
            documents: normalized passage records, each expected to have at
                least "id" and "passage"; "is_selected", "language",
                "query_id", "query_type", "metadata" are used if present.

        Returns:
            List of chunk dicts, one or more per input document. Documents
            with empty/missing passage text are skipped, not crashed on.
        """
        chunks: List[Dict[str, Any]] = []

        for doc in documents:
            passage = doc.get("passage", "") or ""
            if not isinstance(passage, str) or not passage.strip():
                continue

            doc_id = doc.get("id", "")
            language = doc.get("language", self.language)
            is_selected = bool(doc.get("is_selected", False))
            base_metadata = {
                **(doc.get("metadata") or {}),
                "original_id": doc_id,
                "query": doc.get("query", ""),
                "answer": doc.get("answer", ""),
                "is_selected": is_selected,
                "query_type": doc.get("query_type", "unknown"),
            }

            if is_selected:
                # Protected: never split the gold-answer passage.
                chunks.append({
                    "text": passage.strip(),
                    "chunk_id": doc_id,
                    "type": "metadata",
                    "language": language,
                    "chunk_strategy": "metadata_protected",
                    "metadata": base_metadata,
                })
            else:
                sub_chunks = self.semantic_chunker.chunk(passage, metadata=base_metadata)
                for i, sc in enumerate(sub_chunks):
                    chunks.append({
                        "text": sc["text"],
                        "chunk_id": f"{doc_id}_{i}",
                        "type": "metadata",
                        "language": language,
                        "chunk_strategy": "metadata_delegated",
                        "metadata": base_metadata,
                    })

        logger.debug(f"Created {len(chunks)} metadata-aware chunks from {len(documents)} documents")
        return chunks
