"""
Pipeline Orchestrator

Coordinates the entire RAG pipeline from audio to answer.
"""

import asyncio
import time
from typing import Optional, Tuple
import logging

from .schemas import (
    AudioRequest,
    AudioResponse,
    QueryRequest,
    QueryResponse,
    Evidence,
    ProcessingStage,
)
from . import stt, retrieval, generation, guardrails
from .metrics import MetricsCollector
from config import settings

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


async def process_audio(request: AudioRequest) -> AudioResponse:
    """
    Process audio file through STT pipeline.
    
    Args:
        request: AudioRequest containing audio data
        
    Returns:
        AudioResponse with transcript and latency
        
    Raises:
        PipelineError: If audio processing fails
    """
    start_time = time.perf_counter()
    
    try:
        # Process audio through Groq Whisper
        transcript, language = await stt.transcribe_audio(
            audio_data=request.audio_data,
            format=request.format,
            language=request.language,
        )
        
        stt_latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Collect metrics
        MetricsCollector.record(
            stage=ProcessingStage.STT,
            latency_ms=stt_latency_ms,
        )
        
        logger.info(f"STT completed in {stt_latency_ms:.2f}ms")
        
        return AudioResponse(
            transcript=transcript,
            language=language,
            stt_latency_ms=stt_latency_ms,
        )
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}", exc_info=True)
        raise PipelineError(f"Audio processing failed: {e}")


async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Process text query through full RAG pipeline.
    
    Args:
        request: QueryRequest containing query text
        
    Returns:
        QueryResponse with answer, evidence, and latency
        
    Raises:
        PipelineError: If query processing fails
    """
    start_time = time.perf_counter()
    
    try:
        # Step 1: Normalize query
        normalized_query = retrieval.normalize_query(request.query)
        
        # Step 2: Parallel retrieval (Qdrant + BM25)
        retrieval_start = time.perf_counter()
        qdrant_task = asyncio.create_task(
            retrieval.search_qdrant(normalized_query)
        )
        bm25_task = asyncio.create_task(
            retrieval.search_bm25(normalized_query)
        )
        
        (qdrant_evidence, qdrant_latency_ms), (bm25_evidence, bm25_latency_ms) = await asyncio.gather(
            qdrant_task, bm25_task
        )

        # Step 3: Fuse results using RRF
        fused_evidence = retrieval.fuse_results(
            qdrant_evidence, bm25_evidence, k=5
        )
        
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
        
        # Step 4: Evidence gate check
        gate_result = await guardrails.check_evidence(
            query=normalized_query,
            evidence=fused_evidence,
        )
        
        if not gate_result.passed:
            logger.warning(f"Evidence gate failed: {gate_result.reason}")
            return QueryResponse(
                query=request.query,
                answer="I don't have enough information to answer this question.",
                evidence=[e.passage for e in fused_evidence],
                confidence=gate_result.confidence,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                language=request.language,
            )
        
        # Step 5: Build context for LLM
        context = generation.build_context(
            query=normalized_query,
            evidence=fused_evidence,
        )
        
        # Step 6: Generate answer using Groq LLM
        generation_start = time.perf_counter()
        generated_text = await generation.generate_answer(
            query=normalized_query,
            context=context,
            language=request.language,
        )
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000
        
        # Step 7: Validate response
        validation_start = time.perf_counter()
        validation_result = await guardrails.validate_response(
            query=normalized_query,
            answer=generated_text,
            evidence=fused_evidence,
        )
        validation_latency_ms = (time.perf_counter() - validation_start) * 1000
        
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Collect metrics
        MetricsCollector.record(ProcessingStage.NORMALIZATION, latency_ms=0)
        MetricsCollector.record(ProcessingStage.QDRANT_SEARCH, latency_ms=qdrant_latency_ms)
        MetricsCollector.record(ProcessingStage.BM25_SEARCH, latency_ms=bm25_latency_ms)
        MetricsCollector.record(ProcessingStage.RRF_FUSION, latency_ms=0)
        MetricsCollector.record(ProcessingStage.EVIDENCE_GATE, latency_ms=0)
        MetricsCollector.record(ProcessingStage.CONTEXT_BUILDING, latency_ms=0)
        MetricsCollector.record(ProcessingStage.LLM_GENERATION, latency_ms=generation_latency_ms)
        MetricsCollector.record(ProcessingStage.VALIDATION, latency_ms=validation_latency_ms)
        
        logger.info(f"Query processed in {total_latency_ms:.2f}ms")
        
        return QueryResponse(
            query=request.query,
            answer=generated_text,
            evidence=[e.passage for e in fused_evidence],
            confidence=validation_result.confidence if validation_result else 1.0,
            latency_ms=total_latency_ms,
            language=request.language,
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise PipelineError(f"Query processing failed: {e}")
