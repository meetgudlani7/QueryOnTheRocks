"""
Semantic / Token Chunker (Strategy B)

Sliding-window chunking by (approximate) token count with configurable
overlap, so information sitting near a window boundary isn't lost between
chunks. "Token" here is a whitespace-split word — a crude but genuinely
language-agnostic proxy that works uniformly across English and the Indic
scripts in MSMARCO-XI without pulling in a per-language tokenizer.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Sliding-window chunker over whitespace-tokenized text, with overlap."""

    def __init__(
        self,
        min_tokens: int = 120,
        max_tokens: int = 180,
        overlap_tokens: int = 25,
        language: str = "en",
    ):
        if min_tokens < 1 or max_tokens < 1:
            raise ValueError("min_tokens and max_tokens must be >= 1")
        if min_tokens > max_tokens:
            raise ValueError(f"min_tokens ({min_tokens}) must be <= max_tokens ({max_tokens})")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if overlap_tokens >= max_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens}), "
                "or the window would never advance"
            )
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.language = language

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Args:
            text: text to chunk
            metadata: metadata to attach to each chunk

        Returns:
            List of chunk dicts. Empty/whitespace-only input returns [].
            Text at or below max_tokens returns a single whole-text chunk —
            there's no point windowing something that already fits.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        tokens = text.split()
        if len(tokens) <= self.max_tokens:
            return [{
                "text": text.strip(),
                "chunk_id": 0,
                "type": "semantic",
                "language": self.language,
                "metadata": metadata or {},
            }]

        step = self.max_tokens - self.overlap_tokens  # always >= 1, validated above
        chunks: List[Dict[str, Any]] = []
        start = 0
        while start < len(tokens):
            window = tokens[start:start + self.max_tokens]
            chunks.append({
                "text": " ".join(window),
                "chunk_id": len(chunks),
                "type": "semantic",
                "language": self.language,
                "metadata": metadata or {},
            })
            if start + self.max_tokens >= len(tokens):
                break
            start += step

        logger.debug(f"Split {len(tokens)} tokens into {len(chunks)} overlapping windows")
        return chunks
