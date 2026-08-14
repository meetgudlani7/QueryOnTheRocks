"""
FastAPI Main Application

Voice-Enabled RAG System - Entry point for the backend API.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routes import health, audio, query
from apps.api.middleware import timing
from config import configure_logging
from retrieval import embeddings as embedding_engine
from retrieval import qdrant_store, bm25_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Do the slow, one-time setup here instead of paying for it on whichever
    request happens to arrive first: load the embedding model, open the
    (reused) Qdrant connection, and load the persisted BM25 index.
    """
    configure_logging()
    logger.info("Startup: preloading embedding model...")
    try:
        await asyncio.to_thread(embedding_engine.preload)
    except Exception as e:
        # Don't crash the whole API over this — health checks should still
        # work, and the error will surface clearly on the first real query.
        logger.error(f"Embedding model preload failed, will retry lazily on first request: {e}")

    qdrant_store.get_store()
    bm25_store.get_store()
    logger.info("Startup complete")

    yield

    logger.info("Shutdown: closing Qdrant client...")
    await qdrant_store.close()


# Create FastAPI app
app = FastAPI(
    title="Voice-Enabled RAG API",
    description="Production-quality multilingual voice-enabled RAG system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware
app.middleware("http")(timing.timing_middleware)

# Include routers
app.include_router(health.router)
app.include_router(audio.router)
app.include_router(query.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Voice-Enabled RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
