"""
Chunking Orchestrator

Combines all three chunking strategies (sentence-window, token-window,
metadata-adaptive) per passage into one deduplicated set of chunk records
ready for embedding. Deliberately hybrid: no single strategy is trusted
alone.

- Each strategy runs independently and its failure is caught — a bug or
  edge case that breaks one strategy for one passage does not lose that
  passage's other chunks, and does not crash the whole ingest run.
- If every strategy produces nothing usable for a passage (e.g. all raised,
  or the text is a pathological edge case none of them handle), the whole
  passage is indexed as a single fallback chunk rather than silently
  dropped — a chunking bug should degrade retrieval granularity, not
  silently shrink the knowledge base.
- Identical chunk text produced by more than one strategy is stored once,
  tagged with every strategy that produced it — that agreement is itself a
  useful signal, not something to discard.
- A per-passage cap prevents a pathological long passage / small overlap
  step combination from generating an unbounded number of chunks.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Set

from config import settings
from .chunkers.metadata import MetadataChunker
from .chunkers.semantic import SemanticChunker
from .chunkers.sentence import SentenceChunker

logger = logging.getLogger(__name__)


class ChunkingError(Exception):
    """Custom exception for chunking errors."""
    pass


def chunk_documents(
    documents: List[Dict[str, Any]],
    max_chunks_per_passage: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Args:
        documents: normalized passage records (ingestion.normalize output)
        max_chunks_per_passage: safety cap; defaults to settings.CHUNK_MAX_PER_PASSAGE

    Returns:
        Flat list of chunk records — each is a copy of its source passage
        record with "id" and "passage" replaced by the chunk's own id/text,
        and "metadata" enriched with chunk_strategy / chunk_index / etc.

    Raises:
        ChunkingError: if the input produced zero usable chunks overall —
            a real, working pipeline should never do this, so it signals an
            upstream problem rather than silently returning an empty index.
    """
    if not documents:
        return []

    max_chunks_per_passage = max_chunks_per_passage or settings.CHUNK_MAX_PER_PASSAGE

    sentence_chunker = SentenceChunker(
        sentences_per_chunk=settings.CHUNK_SENTENCES_PER_CHUNK,
        overlap_sentences=settings.CHUNK_SENTENCE_OVERLAP,
    )
    semantic_chunker = SemanticChunker(
        min_tokens=settings.CHUNK_MIN_TOKENS,
        max_tokens=settings.CHUNK_MAX_TOKENS,
        overlap_tokens=settings.CHUNK_TOKEN_OVERLAP,
    )
    metadata_chunker = MetadataChunker(semantic_chunker=semantic_chunker)

    global_seen_hashes: Set[str] = set()
    output: List[Dict[str, Any]] = []
    stats = {"passages": 0, "chunks_kept": 0, "chunks_deduped": 0, "passages_empty": 0, "passages_capped": 0, "passages_fallback": 0}

    for doc in documents:
        passage_id = doc.get("id")
        passage_text = doc.get("passage", "") or ""
        if passage_id is None:
            logger.warning("Skipping document with no id")
            continue
        if not isinstance(passage_text, str) or not passage_text.strip():
            stats["passages_empty"] += 1
            continue

        stats["passages"] += 1
        language = doc.get("language", "unknown")

        candidates: List[tuple] = []

        try:
            for c in sentence_chunker.chunk(passage_text):
                candidates.append((c["text"], "sentence"))
        except Exception as e:
            logger.warning(f"Sentence chunking failed for passage {passage_id}: {e}")

        try:
            for c in semantic_chunker.chunk(passage_text):
                candidates.append((c["text"], "semantic"))
        except Exception as e:
            logger.warning(f"Semantic chunking failed for passage {passage_id}: {e}")

        try:
            for c in metadata_chunker.chunk([doc]):
                candidates.append((c["text"], c.get("chunk_strategy", "metadata")))
        except Exception as e:
            logger.warning(f"Metadata chunking failed for passage {passage_id}: {e}")

        if not candidates:
            # Every strategy failed or produced nothing — fall back to the
            # whole passage rather than losing it from the index entirely.
            stats["passages_fallback"] += 1
            candidates = [(passage_text, "fallback_whole_passage")]

        # Dedupe within this passage, merging strategy tags for identical text.
        per_passage: Dict[str, List[str]] = {}
        order: List[str] = []
        for text, strategy in candidates:
            text_norm = " ".join(text.split())
            if not text_norm:
                continue
            if text_norm not in per_passage:
                per_passage[text_norm] = [strategy]
                order.append(text_norm)
            elif strategy not in per_passage[text_norm]:
                per_passage[text_norm].append(strategy)

        kept_for_passage = 0
        for idx, text_norm in enumerate(order):
            if kept_for_passage >= max_chunks_per_passage:
                stats["passages_capped"] += 1
                break

            content_hash = hashlib.sha1(f"{language}|{text_norm.lower()}".encode("utf-8")).hexdigest()
            if content_hash in global_seen_hashes:
                stats["chunks_deduped"] += 1
                continue
            global_seen_hashes.add(content_hash)

            strategies = per_passage[text_norm]
            chunk_record = dict(doc)
            chunk_record["id"] = f"{passage_id}_chunk{idx}"
            chunk_record["passage"] = text_norm
            chunk_record["metadata"] = {
                **(doc.get("metadata") or {}),
                "source_passage_id": passage_id,
                "chunk_index": idx,
                "chunk_strategy": "+".join(strategies),
                "chunk_strategies": strategies,
                "char_count": len(text_norm),
                "approx_token_count": len(text_norm.split()),
            }
            output.append(chunk_record)
            kept_for_passage += 1
            stats["chunks_kept"] += 1

    logger.info(
        f"Chunked {stats['passages']} passages -> {stats['chunks_kept']} chunks "
        f"(deduped={stats['chunks_deduped']}, empty_skipped={stats['passages_empty']}, "
        f"capped={stats['passages_capped']}, fallback_whole_passage={stats['passages_fallback']})"
    )

    if not output:
        raise ChunkingError("Chunking produced zero usable chunks from the input passages")

    return output
