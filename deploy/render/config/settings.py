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
    # Model used for the post-generation content-safety check (roadmap
    # Phase 23), replacing a fixed 5-keyword regex list that was both
    # trivially bypassable and prone to false-positiving on legitimate
    # content (e.g. a history question mentioning "violence"). None means
    # "reuse GROQ_LLM_MODEL" — deliberately not a separately-guessed model
    # name, since GROQ_LLM_MODEL is already proven working in this
    # deployment; override only if you want a dedicated (e.g. cheaper)
    # moderation model.
    GROQ_MODERATION_MODEL: Optional[str] = None
    
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

    # Scopes Qdrant dense search to {request language, "en"} via a payload
    # filter, instead of searching the full multilingual index every time
    # (ingestion/index.py already stores a "language" field on every point,
    # this just starts using it at query time). Off by default — a wrong
    # language *detection* (STT or an explicit bad hint) would incorrectly
    # exclude the right passage, which is a worse failure than the status
    # quo, so this stays off until an A/B benchmark run
    # (evaluation/queries.jsonl, grouped by language) proves it's neutral
    # or better on recall before flipping the default (roadmap Phase 19).
    RETRIEVAL_LANGUAGE_FILTER: bool = False

    # Reranks the fused Qdrant+BM25 candidates with a cross-encoder before
    # the evidence gate (roadmap Phase 20). Off by default for the same
    # reason as above: proven-neutral-or-better on the benchmark before the
    # default flips, since a broken/unhelpful reranker would silently
    # degrade every answer's evidence quality.
    RERANKING_ENABLED: bool = False
    # cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 is trained specifically on
    # mMARCO (multilingual MS MARCO passage ranking) — this corpus *is*
    # MS MARCO-XI, translated MS MARCO, so this reranker's training
    # distribution is an unusually close match to this system's actual
    # data, not a generic off-the-shelf choice.
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # Candidates pulled from fusion before reranking trims down to
    # MAX_CONTEXT_CHUNKS. Must be wider than MAX_CONTEXT_CHUNKS or there's
    # nothing for the reranker to actually re-rank.
    RERANK_CANDIDATE_K: int = 20
    
    # Embedding configuration
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "auto"  # auto | cpu | cuda | mps
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MAX_SEQ_LENGTH: int = 256

    # torch | onnx — which runtime encodes text into vectors (retrieval/embeddings.py).
    # Default "torch" preserves exact current behavior (SentenceTransformer,
    # unchanged). "onnx" runs the identical weights (see
    # scripts/export_onnx_embedder.py + scripts/verify_onnx_embedder.py,
    # verified numerically equivalent: cosine_sim=1.0, max_abs_diff=0.0
    # across English/Hindi/empty/long-truncated test cases) through
    # onnxruntime + tokenizers instead of torch + sentence-transformers +
    # transformers — ~600MB lighter at runtime, no model-quality change.
    # Opt-in, matching this file's other feature-flag defaults, so nothing
    # about today's deployments changes unless explicitly configured.
    EMBEDDING_BACKEND: str = "torch"
    EMBEDDING_ONNX_DIR: str = "data/onnx_embedder"

    # Coalesces concurrent embed_query() calls arriving within a short
    # window into one batched model.encode() call — GPU/MPS inference
    # throughput benefits significantly from batching over one-vector-at-
    # a-time calls (roadmap Phase 22). Mathematically behavior-preserving
    # when correct (identical _encode_sync() call, just grouped
    # differently — see retrieval/embeddings.py's _EmbeddingBatcher and
    # its equivalence tests). Off by default: this is the most novel
    # concurrency primitive in this phase (a custom async batching queue),
    # and embed_query() sits on the hot path for every dense search and
    # every groundedness check, so a bug here has an unusually wide blast
    # radius — enable only after load-testing in your own environment.
    EMBEDDING_MICROBATCH_ENABLED: bool = False
    EMBEDDING_MICROBATCH_WINDOW_MS: int = 15
    EMBEDDING_MICROBATCH_MAX_SIZE: int = 32

    # Chunking (ingestion/chunk.py — hybrid: sentence + token-window + metadata-adaptive)
    CHUNK_SENTENCES_PER_CHUNK: int = 3
    CHUNK_SENTENCE_OVERLAP: int = 1
    CHUNK_MIN_TOKENS: int = 120
    CHUNK_MAX_TOKENS: int = 180
    CHUNK_TOKEN_OVERLAP: int = 25
    CHUNK_MAX_PER_PASSAGE: int = 20

    # Guardrails
    # Normalized against the max possible RRF score (2/(RRF_CONSTANT+1), i.e.
    # a document ranked #1 in both Qdrant and BM25) — 1.0 is a theoretical
    # perfect match, not a raw RRF score.
    #
    # Calibrated (roadmap Phase 18, scripts/calibrate_guardrails.py) via a
    # live 3x3 grid sweep against evaluation/queries.jsonl (10 stratified
    # queries per combination, real Qdrant + Groq calls). The measured
    # winner pairs a *looser* retrieval gate with a *stricter* groundedness
    # check below (0.05 / 0.5) — combined_score=0.65 vs. 0.55 for the prior
    # untuned default (0.15 / 0.35), which scored worse than 5 of the 9
    # combinations tested. See data/guardrail_calibration_results.json for
    # the full sweep. Caveat: 10 queries/combination is a small sample
    # (refusal_accuracy in particular is computed from ~6 refusal-category
    # queries per combination) — re-run with a higher
    # --max-queries-per-type before treating this as final for a real
    # deployment, but it's a measured improvement over an unvalidated guess.
    MIN_RETRIEVAL_SCORE: float = 0.05
    MAX_CONTEXT_CHUNKS: int = 5
    MAX_ANSWER_TOKENS: int = 150

    # Cosine similarity floor between a generated answer and its cited
    # evidence (see pipeline/guardrails.py's _groundedness_similarity) — an
    # independent check that a "grounded: true" claim is actually supported
    # by the passage cited, not just that the citation ID is real.
    # Calibrated alongside MIN_RETRIEVAL_SCORE above — see that comment.
    MIN_GROUNDEDNESS_SIMILARITY: float = 0.50

    # Bounds concurrent in-flight Groq API calls (roadmap Phase 22) so a
    # traffic burst degrades to "briefly waits its turn" instead of
    # tripping Groq's own per-account rate limits across every concurrent
    # request at once. Separate limits for STT and LLM since they're
    # different Groq models/endpoints with independent quotas. Sized
    # conservatively for an on-demand tier; raise if your plan allows more.
    GROQ_LLM_MAX_CONCURRENT: int = 5
    GROQ_STT_MAX_CONCURRENT: int = 5

    # API key auth (roadmap Phase 23) — opt-in: empty (default) disables
    # auth entirely, matching today's behavior exactly. Comma-separated so
    # multiple keys (e.g. one per client/environment) can be issued and
    # revoked independently without sharing one secret.
    API_KEYS: str = ""

    # In-process token-bucket rate limiting per client (API key if
    # present, else IP) — roadmap Phase 23. Safe to default on: nothing in
    # this repo's tests/scripts talks to the API over real HTTP (they call
    # pipeline functions directly), so this can't break any existing
    # automated check. 0 disables entirely.
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60

    # Explicit CORS allow-list (roadmap Phase 23), replacing a wildcard
    # origin combined with allow_credentials=True — permissive defaults
    # that are fine for local development but shouldn't ship as-is.
    # Comma-separated; defaults to the local dev frontend only.
    CORS_ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Hard ceiling on concurrent in-flight requests to /api/* routes
    # (apps/api/middleware/concurrency.py) — protects the shared,
    # process-wide embedding model and Qdrant/BM25 connections from
    # unbounded pile-up. A request that can't be admitted within
    # CONCURRENCY_QUEUE_TIMEOUT_MS gets a clean 503 instead of silently
    # degrading every other in-flight request's latency. 0 disables the
    # ceiling entirely (unbounded, today's behavior).
    MAX_CONCURRENT_REQUESTS: int = 20
    CONCURRENCY_QUEUE_TIMEOUT_MS: int = 2000

    # Stage timeouts
    STT_TIMEOUT_MS: int = 5000
    # The guide's aspirational 500ms is far below measured reality: a warm
    # Qdrant Cloud round-trip from this dev setup runs ~300-750ms. Setting
    # the hard timeout at the aspirational number would make degrading to
    # BM25-only the common case instead of the exception. 3000ms gives
    # genuine headroom above observed normal latency while still catching
    # real hangs; revisit once deployed nearer the Qdrant region.
    RETRIEVAL_TIMEOUT_MS: int = 3000
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

    # Optional per-stage tracing (roadmap Phase 24, see config/tracing.py).
    # Off by default — zero behavior/dependency change unless explicitly
    # enabled. Exports to the console when on but no OTLP endpoint is set;
    # set OTEL_EXPORTER_OTLP_ENDPOINT to export to a real backend.
    TRACING_ENABLED: bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Create settings instance
settings = Settings()
