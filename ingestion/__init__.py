"""
Ingestion Module

Handles dataset loading, processing, and indexing.
"""

from .download import download_dataset
from .normalize import normalize_data
from .chunk import chunk_documents
from .embed import generate_embeddings
from .index import build_index

__all__ = ["download_dataset", "normalize_data", "chunk_documents", "generate_embeddings", "build_index"]
