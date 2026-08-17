"""
Benchmark Module

Runs benchmark tests on the RAG system.
"""

import asyncio
import io
import struct
import time
import wave
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json

from pipeline import process_query, process_audio, QueryRequest
from pipeline.schemas import AudioRequest, QueryResponse
from .metrics import ANSWERABLE_TYPES, calculate_f1, F1_MATCH_THRESHOLD

logger = logging.getLogger(__name__)


class BenchmarkError(Exception):
    """Custom exception for benchmark errors."""
    pass


async def run_benchmark(
    queries_file: Path = Path("evaluation/queries.jsonl"),
    max_queries: int = 10,
    request_delay_s: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run benchmark on a set of queries.

    Args:
        queries_file: Path to JSONL file with queries
        max_queries: Maximum number of queries to run
        request_delay_s: pause before each query after the first. A quality
            report needs the model to actually be reached, not rate-limited
            (see scripts/run_benchmark.py) — this is a benchmark-harness
            pacing concern, not a change to the pipeline's own retry policy.

    Returns:
        Tuple of (results, summary)

    Raises:
        BenchmarkError: If benchmark fails
    """
    try:
        # Load queries
        queries = _load_queries(queries_file, max_queries)

        if not queries:
            raise BenchmarkError("No queries loaded")

        logger.info(f"Running benchmark with {len(queries)} queries")

        results = []
        start_time = time.perf_counter()

        for i, query_data in enumerate(queries, 1):
            if request_delay_s and i > 1:
                await asyncio.sleep(request_delay_s)
            logger.info(f"Processing query {i}/{len(queries)}: {query_data.get('query', '')[:50]}...")

            request = QueryRequest(
                query=query_data.get("query", ""),
                language=query_data.get("language", "en"),
            )
            
            query_type = query_data.get("type", "factoid")
            expected_answer = query_data.get("answer", "")

            try:
                response = await process_query(request)

                result = {
                    "query_id": query_data.get("id", i),
                    "query": query_data.get("query", ""),
                    "type": query_type,
                    "answer": response.answer,
                    "expected_answer": expected_answer,
                    "confidence": response.confidence,
                    "grounded": response.grounded,
                    "latency_ms": response.latency_ms,
                    "evidence": response.evidence,
                    "evidence_ids": response.evidence_ids,
                    "evidence_count": len(response.evidence),
                    "relevant_chunk_ids": query_data.get("relevant_chunk_ids", []),
                    "passed": _check_answer(response.grounded, response.answer, expected_answer, query_type),
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Query {i} failed: {e}")
                results.append({
                    "query_id": query_data.get("id", i),
                    "query": query_data.get("query", ""),
                    "type": query_type,
                    "error": str(e),
                    "passed": False,
                })
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Calculate summary
        summary = _calculate_summary(results, total_time)
        
        return results, summary
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        raise BenchmarkError(f"Benchmark failed: {e}")


def _silent_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """A minimal valid WAV file (silence) for latency probing — see run_voice_latency_benchmark."""
    n_frames = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return buf.getvalue()


async def run_voice_latency_benchmark(n: int = 30, request_delay_s: float = 0.0) -> List[Dict[str, Any]]:
    """
    Measures real STT and end-to-end voice latency by running process_audio
    (STT -> full RAG pipeline) n times against a fixed short audio sample.

    Content is irrelevant here — this measures actual Groq/network/pipeline
    timing, not transcription accuracy (grounding quality is what
    run_benchmark's text queries cover). Every latency value is a real
    measured round trip, never estimated or summed from separate calls.

    Args:
        request_delay_s: pause before each run after the first — see
            run_benchmark's request_delay_s for why.
    """
    audio_bytes = _silent_wav_bytes()
    results = []

    for i in range(1, n + 1):
        if request_delay_s and i > 1:
            await asyncio.sleep(request_delay_s)
        start = time.perf_counter()
        try:
            response = await process_audio(AudioRequest(audio_data=audio_bytes, format="audio/wav", language="en"))
            total_latency_ms = (time.perf_counter() - start) * 1000
            results.append({
                "run": i,
                "stt_latency_ms": response.stt_latency_ms,
                "rag_latency_ms": response.rag_latency_ms,
                "total_latency_ms": total_latency_ms,
            })
        except Exception as e:
            logger.error(f"Voice latency run {i}/{n} failed: {e}")
            results.append({"run": i, "error": str(e)})

    return results


def _load_queries(file_path: Path, max_queries: int) -> List[Dict[str, Any]]:
    """Load queries from JSONL file."""
    queries = []
    
    try:
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        queries.append(json.loads(line))
                        if max_queries and len(queries) >= max_queries:
                            break
        else:
            logger.warning(f"Queries file not found: {file_path}")
    except Exception as e:
        logger.error(f"Failed to load queries: {e}")
    
    return queries


def _check_answer(grounded: bool, answer: str, expected: str, query_type: str) -> bool:
    """
    Category-aware pass/fail check.

    Refusal categories (unanswerable/off-topic/injection) pass when the
    pipeline did NOT ground an answer. Answerable categories pass when the
    pipeline grounded an answer with enough token overlap (F1, SQuAD-style)
    against the expected answer — an LLM restating a fact in its own words
    almost never survives an exact-substring check, so that check was
    mostly measuring paraphrasing, not correctness.
    """
    if query_type not in ANSWERABLE_TYPES:
        return not grounded

    if not expected:
        return grounded

    return grounded and calculate_f1(answer, expected) >= F1_MATCH_THRESHOLD


def _calculate_summary(results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
    """Calculate benchmark summary."""
    total_queries = len(results)
    passed_queries = sum(1 for r in results if r.get("passed", False))
    
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    return {
        "total_queries": total_queries,
        "passed_queries": passed_queries,
        "failed_queries": total_queries - passed_queries,
        "success_rate": (passed_queries / total_queries) * 100 if total_queries > 0 else 0,
        "total_time_ms": total_time,
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min(latencies) if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
    }
