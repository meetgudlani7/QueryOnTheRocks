"""
Pipeline Orchestrator

Coordinates the entire RAG pipeline from audio to answer.
"""

import asyncio
import time
import uuid
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
from .generation import GenerationError
from .metrics import MetricsCollector
from config import settings
from config.logging import request_id_var

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


async def process_audio(request: AudioRequest) -> AudioResponse:
    """
    Process audio file through STT, then through the full RAG pipeline so
    voice questions get real answers, not just a transcript.

    Args:
        request: AudioRequest containing audio data

    Returns:
        AudioResponse with transcript, RAG answer, and latency breakdown

    Raises:
        PipelineError: If audio processing fails
    """
    request_id = str(uuid.uuid4())
    token = request_id_var.set(request_id)
    start_time = time.perf_counter()

    try:
        transcript, language = await stt.transcribe_audio(
            audio_data=request.audio_data,
            format=request.format,
            language=request.language,
        )

        stt_latency_ms = (time.perf_counter() - start_time) * 1000

        MetricsCollector.record(
            stage=ProcessingStage.STT,
            latency_ms=stt_latency_ms,
        )

        logger.info(f"STT completed in {stt_latency_ms:.2f}ms")

        rag_start = time.perf_counter()
        query_response = await process_query(QueryRequest(query=transcript, language=language))
        rag_latency_ms = (time.perf_counter() - rag_start) * 1000

        return AudioResponse(
            transcript=transcript,
            language=language,
            stt_latency_ms=stt_latency_ms,
            answer=query_response.answer,
            evidence=query_response.evidence,
            confidence=query_response.confidence,
            grounded=query_response.grounded,
            rag_latency_ms=rag_latency_ms,
            request_id=request_id,
        )

    except Exception as e:
        logger.error(f"Audio processing failed: {e}", exc_info=True)
        raise PipelineError(f"Audio processing failed: {e}")
    finally:
        request_id_var.reset(token)


def _safe_response(request: QueryRequest, request_id: str, answer: str, confidence: float, start_time: float) -> QueryResponse:
    """Shared constructor for the various safe-refusal/safe-failure responses below."""
    return QueryResponse(
        query=request.query,
        answer=answer,
        evidence=[],
        evidence_ids=[],
        confidence=confidence,
        grounded=False,
        latency_ms=(time.perf_counter() - start_time) * 1000,
        language=request.language,
        request_id=request_id,
    )


async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Process text query through full RAG pipeline.

    Args:
        request: QueryRequest containing query text

    Returns:
        QueryResponse with answer, evidence, and latency

    Raises:
        PipelineError: If query processing fails in a way that has no safe
            fallback (e.g. retrieval infra entirely unavailable)
    """
    request_id = str(uuid.uuid4())
    token = request_id_var.set(request_id)
    start_time = time.perf_counter()

    try:
        # Step 1: Normalize query
        normalized_query = retrieval.normalize_query(request.query)

        # Step 2: Parallel retrieval (Qdrant + BM25). Each is allowed to
        # fail independently — return_exceptions=True means one backend
        # being down degrades to single-source retrieval instead of
        # failing the whole request.
        retrieval_start = time.perf_counter()
        qdrant_task = asyncio.create_task(retrieval.search_qdrant(normalized_query))
        bm25_task = asyncio.create_task(retrieval.search_bm25(normalized_query))

        qdrant_result, bm25_result = await asyncio.gather(qdrant_task, bm25_task, return_exceptions=True)

        if isinstance(qdrant_result, BaseException):
            logger.warning(f"Qdrant search failed, continuing with BM25 only: {qdrant_result}")
            qdrant_evidence, qdrant_latency_ms = [], 0.0
        else:
            qdrant_evidence, qdrant_latency_ms = qdrant_result

        if isinstance(bm25_result, BaseException):
            logger.warning(f"BM25 search failed, continuing with Qdrant only: {bm25_result}")
            bm25_evidence, bm25_latency_ms = [], 0.0
        else:
            bm25_evidence, bm25_latency_ms = bm25_result

        if isinstance(qdrant_result, BaseException) and isinstance(bm25_result, BaseException):
            logger.error("Both Qdrant and BM25 retrieval failed")
            return _safe_response(
                request, request_id,
                "The knowledge service is temporarily unavailable.",
                0.0, start_time,
            )

        # Step 3: Fuse results using RRF
        fused_evidence = retrieval.fuse_results(qdrant_evidence, bm25_evidence, k=settings.MAX_CONTEXT_CHUNKS)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        # Step 4: Evidence gate check
        gate_result = await guardrails.check_evidence(query=normalized_query, evidence=fused_evidence)

        if not gate_result.passed:
            logger.warning(f"Evidence gate failed: {gate_result.reason}")
            return QueryResponse(
                query=request.query,
                answer="I couldn't find enough information in the knowledge base to answer that.",
                evidence=[e.passage for e in fused_evidence],
                evidence_ids=[e.document_id for e in fused_evidence],
                confidence=gate_result.confidence,
                grounded=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                language=request.language,
                request_id=request_id,
            )

        # Step 5: Build context for LLM
        context = generation.build_context(fused_evidence)
        valid_evidence_ids = [e.document_id for e in fused_evidence]

        # Step 6: Generate answer using Groq LLM
        generation_start = time.perf_counter()
        try:
            generation_result = await generation.generate_answer(
                query=normalized_query,
                context=context,
                valid_evidence_ids=valid_evidence_ids,
                language=request.language,
            )
        except GenerationError as e:
            logger.error(f"Generation failed: {e}")
            return _safe_response(
                request, request_id,
                "Relevant information was found, but the answer service is temporarily unavailable.",
                0.0, start_time,
            )
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        # Step 7: Validate response — and actually act on the result. An
        # earlier version computed this and then returned the answer
        # regardless of whether it passed, which made the whole guardrail
        # a no-op.
        validation_start = time.perf_counter()
        validation_result = await guardrails.validate_response(
            generation_result=generation_result,
            evidence=fused_evidence,
        )
        validation_latency_ms = (time.perf_counter() - validation_start) * 1000

        if not validation_result.passed:
            logger.warning(f"Response validation failed: {validation_result.reason}")
            return QueryResponse(
                query=request.query,
                answer="I found some information but couldn't produce a fully grounded answer. Please try rephrasing your question.",
                evidence=[e.passage for e in fused_evidence],
                evidence_ids=[e.document_id for e in fused_evidence],
                confidence=0.0,
                grounded=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                language=request.language,
                request_id=request_id,
            )

        total_latency_ms = (time.perf_counter() - start_time) * 1000

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
            answer=generation_result.answer,
            evidence=[e.passage for e in fused_evidence],
            evidence_ids=[e.document_id for e in fused_evidence],
            confidence=validation_result.confidence,
            grounded=generation_result.grounded,
            latency_ms=total_latency_ms,
            language=request.language,
            request_id=request_id,
        )

    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise PipelineError(f"Query processing failed: {e}")
    finally:
        request_id_var.reset(token)
