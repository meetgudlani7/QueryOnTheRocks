"""
BM25 Keyword Search Store

Handles keyword search using BM25 algorithm.
"""

import time
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import heapq
import math
from collections import defaultdict
import re

from config import settings

logger = logging.getLogger(__name__)


class BM25Error(Exception):
    """Custom exception for BM25 errors."""
    pass


class BM25Store:
    """
    In-memory BM25 keyword search index.
    
    Note: In production, consider using a dedicated library like rank_bm25
    or a database with BM25 support.
    """
    
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
        self._initialized: bool = False
        
    async def search(
        self,
        query: str,
        k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query using BM25.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of result documents with scores
            
        Raises:
            BM25Error: If search fails
        """
        start_time = time.perf_counter()
        
        try:
            if not self._initialized:
                raise BM25Error("BM25 index not initialized")
            
            # Tokenize query
            query_terms = self._tokenize(query)
            
            # Calculate BM25 scores for each document
            scores = []
            for doc_id in range(self.total_docs):
                score = 0.0
                for term in query_terms:
                    if term in self.inverted_index and doc_id in self.inverted_index[term]:
                        tf = self.inverted_index[term][doc_id]
                        idf = self._idf(term)
                        score += self._bm25_score(tf, idf, self.doc_lengths[doc_id])
                
                if score > 0:
                    scores.append((-score, doc_id))  # Negative for min-heap
            
            # Get top k results
            top_k = heapq.nsmallest(k, scores)
            
            # Transform results
            results = []
            for neg_score, doc_id in top_k:
                doc = self.documents[doc_id]
                results.append({
                    "passage": doc.get("passage", ""),
                    "score": -neg_score,
                    "document_id": doc.get("id", str(doc_id)),
                    "language": doc.get("language", "en"),
                    "metadata": doc.get("metadata", {}),
                })
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"BM25 search completed in {latency_ms:.2f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}", exc_info=True)
            raise BM25Error(f"BM25 search failed: {e}")
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> None:
        """
        Add documents to the BM25 index.
        
        Args:
            documents: List of documents to index
            
        Raises:
            BM25Error: If adding documents fails
        """
        try:
            start_index = self.total_docs
            
            # Add documents
            for doc in documents:
                doc_id = start_index + len(self.documents)
                
                # Tokenize document
                text = doc.get("passage", "")
                tokens = self._tokenize(text)
                doc_length = len(tokens)
                
                # Update inverted index
                term_freq = defaultdict(int)
                for token in tokens:
                    term_freq[token] += 1
                
                for term, tf in term_freq.items():
                    self.inverted_index[term][doc_id] = tf
                    self.doc_freq[term] += 1
                
                self.documents.append(doc)
                self.doc_lengths.append(doc_length)
                self.total_docs += 1
            
            # Recalculate average document length
            if self.doc_lengths:
                self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
            
            self._initialized = True
            
            logger.info(f"Added {len(documents)} documents to BM25 index")
            
        except Exception as e:
            logger.error(f"Adding documents to BM25 failed: {e}", exc_info=True)
            raise BM25Error(f"Adding documents failed: {e}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple tokenizer for demonstration
        # In production, use a more sophisticated tokenizer
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]  # Remove single-character tokens
    
    def _idf(self, term: str) -> float:
        """
        Calculate Inverse Document Frequency.
        
        Args:
            term: Query term
            
        Returns:
            IDF score
        """
        n = self.total_docs
        df = self.doc_freq.get(term, 0)
        
        if df == 0:
            return 0.0
        
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)
    
    def _bm25_score(self, tf: int, idf: float, doc_length: int) -> float:
        """
        Calculate BM25 score for a term in a document.
        
        Args:
            tf: Term frequency in document
            idf: Inverse document frequency
            doc_length: Document length
            
        Returns:
            BM25 score
        """
        k1 = 1.5  # Term frequency saturation parameter
        b = 0.75  # Document length normalization parameter
        
        if self.avg_doc_length == 0:
            return 0.0
        
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))

        return idf * (numerator / denominator)

    def save(self, path: Optional[str] = None) -> None:
        """
        Persist the index to disk so the API doesn't have to rebuild it from
        the dataset on every process restart — the guide is explicit that
        the index should be built once offline and loaded, not recomputed
        per boot.
        """
        path_obj = Path(path or settings.BM25_INDEX_PATH)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "documents": self.documents,
            "inverted_index": dict(self.inverted_index),
            "doc_freq": dict(self.doc_freq),
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "total_docs": self.total_docs,
        }
        tmp = path_obj.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path_obj)  # atomic on POSIX: no half-written index file on crash
            logger.info(f"Saved BM25 index ({self.total_docs} docs) to {path_obj}")
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise BM25Error(f"Failed to save BM25 index to {path_obj}: {e}") from e

    @classmethod
    def load(cls, path: Optional[str] = None) -> "BM25Store":
        """
        Load a previously-saved index. If none exists yet, returns an empty,
        uninitialized store rather than raising — the process should still
        boot (e.g. so /health responds) even before the first index build;
        search() will fail clearly with BM25Error until one is built.
        """
        store = cls()
        path_obj = Path(path or settings.BM25_INDEX_PATH)
        if not path_obj.exists():
            logger.warning(
                f"No BM25 index found at {path_obj} — run scripts/build_index.py first. "
                "BM25 search will raise until then."
            )
            return store

        try:
            with open(path_obj, "rb") as f:
                state = pickle.load(f)
            store.documents = state["documents"]
            store.inverted_index = defaultdict(dict, state["inverted_index"])
            store.doc_freq = defaultdict(int, state["doc_freq"])
            store.doc_lengths = state["doc_lengths"]
            store.avg_doc_length = state["avg_doc_length"]
            store.total_docs = state["total_docs"]
            store._initialized = store.total_docs > 0
            logger.info(f"Loaded BM25 index ({store.total_docs} docs) from {path_obj}")
        except (OSError, pickle.PickleError, KeyError) as e:
            logger.error(f"Failed to load BM25 index from {path_obj}, starting empty: {e}")
        return store


# --- module-level singleton -------------------------------------------------
# pipeline/retrieval.py calls `bm25_store.search(...)` as a plain module
# function (not a class method) — these top-level functions provide that
# interface, backed by one shared BM25Store instance loaded once from disk.

_default_store: Optional[BM25Store] = None


def get_store() -> BM25Store:
    global _default_store
    if _default_store is None:
        _default_store = BM25Store.load()
    return _default_store


def set_store(store: BM25Store) -> None:
    """Used by scripts/build_index.py to install a freshly-built index without a disk round-trip."""
    global _default_store
    _default_store = store


async def search(query: str, k: int = 20) -> List[Dict[str, Any]]:
    return await get_store().search(query, k=k)


async def add_documents(documents: List[Dict[str, Any]]) -> None:
    await get_store().add_documents(documents)
