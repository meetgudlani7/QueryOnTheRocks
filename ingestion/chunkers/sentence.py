"""
Sentence Chunker (Strategy A)

Groups neighboring sentences into overlapping windows. Good for precise,
short evidence — useful for factual questions where the answer sits in one
or two adjacent sentences.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Standard Latin sentence terminators plus the Devanagari danda marks (।॥),
# since MSMARCO-XI's translated passages are Indic-script text that can use
# either convention depending on how the translation model punctuated it.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")


class SentenceChunker:
    """Chunks text into overlapping windows of N neighboring sentences."""

    def __init__(
        self,
        sentences_per_chunk: int = 3,
        overlap_sentences: int = 1,
        language: str = "en",
    ):
        if sentences_per_chunk < 1:
            raise ValueError("sentences_per_chunk must be >= 1")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        if overlap_sentences >= sentences_per_chunk:
            raise ValueError(
                f"overlap_sentences ({overlap_sentences}) must be < sentences_per_chunk "
                f"({sentences_per_chunk}), or every window would be identical"
            )
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.language = language

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Args:
            text: text to chunk
            metadata: metadata to attach to each chunk

        Returns:
            List of chunk dicts. Empty/whitespace-only input returns [].
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        step = self.sentences_per_chunk - self.overlap_sentences  # always >= 1, validated above
        chunks: List[Dict[str, Any]] = []

        start = 0
        while start < len(sentences):
            window = sentences[start:start + self.sentences_per_chunk]
            if not window:
                break
            chunk_text = " ".join(window)
            chunks.append({
                "text": chunk_text,
                "chunk_id": len(chunks),
                "type": "sentence",
                "language": self.language,
                "metadata": metadata or {},
            })
            if start + self.sentences_per_chunk >= len(sentences):
                break
            start += step

        logger.debug(f"Split {len(sentences)} sentences into {len(chunks)} overlapping chunks")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Splits on sentence-ending punctuation. Text with no recognizable
        sentence boundary (no periods/danda marks — e.g. a list fragment or
        a run-on clause) falls back to treating the whole text as one
        "sentence" rather than producing zero output.
        """
        if not isinstance(text, str) or not text.strip():
            return []
        parts = _SENTENCE_SPLIT_RE.split(text.strip())
        parts = [p.strip() for p in parts if p.strip()]
        return parts
