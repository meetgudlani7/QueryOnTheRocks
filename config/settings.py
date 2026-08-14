"""
Application Settings

Manages configuration from environment variables and defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    # Application settings
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = True
    
    # Groq configuration
    GROQ_API_KEY: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    GROQ_STT_MODEL: str = "whisper-large-v3-turbo"
    GROQ_LLM_MODEL: str = "llama-3.1-70b-versatile"
    
    # Qdrant configuration
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    QDRANT_COLLECTION: str = "msmarco_xi"
    
    # Processing limits
    MAX_CONTEXT_LENGTH: int = 4096
    MAX_RESPONSE_LENGTH: int = 512
    MAX_AUDIO_DURATION: int = 60  # seconds
    
    # Performance targets
    TARGET_LATENCY_MS: float = 200.0
    
    # Dataset configuration
    DATASET_NAME: str = "ai4bharat/MSMARCO-XI"
    DATASET_SPLIT: str = "train"
    DATASET_LIMIT: Optional[int] = None
    DATASET_CACHE_DIR: str = "data/msmarco_xi"
    
    # Retrieval configuration
    QDRANT_SEARCH_K: int = 20
    BM25_SEARCH_K: int = 20
    FUSION_K: int = 5
    RRF_CONSTANT: int = 60
    
    # Embedding configuration
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "auto"  # auto | cpu | cuda | mps
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MAX_SEQ_LENGTH: int = 256

    # Chunking (ingestion/chunk.py — hybrid: sentence + token-window + metadata-adaptive)
    CHUNK_SENTENCES_PER_CHUNK: int = 3
    CHUNK_SENTENCE_OVERLAP: int = 1
    CHUNK_MIN_TOKENS: int = 120
    CHUNK_MAX_TOKENS: int = 180
    CHUNK_TOKEN_OVERLAP: int = 25
    CHUNK_MAX_PER_PASSAGE: int = 20

    # Guardrails
    MIN_RETRIEVAL_SCORE: float = 0.3
    MAX_CONTEXT_CHUNKS: int = 5
    MAX_ANSWER_TOKENS: int = 150

    # Stage timeouts
    STT_TIMEOUT_MS: int = 5000
    RETRIEVAL_TIMEOUT_MS: int = 500
    LLM_TIMEOUT_MS: int = 5000

    # Dataset streaming
    # Each MSMARCO-XI shard is a single target language (see ingestion/download.py
    # module docstring) — this many distinct shards are downloaded and sampled to
    # get real multilingual coverage. Each shard is ~3.7GB, one-time, cached.
    DATASET_NUM_SHARDS: int = 3
    DATASET_SHUFFLE_BUFFER_SIZE: int = 500
    DATASET_DOWNLOAD_TIMEOUT_S: int = 1800

    # Index storage
    QDRANT_UPSERT_BATCH_SIZE: int = 128
    BM25_INDEX_PATH: str = "data/bm25_index.pkl"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Create settings instance
settings = Settings()
