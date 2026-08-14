"""
Benchmark Module

Runs benchmark tests on the RAG system.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json

from pipeline import process_query, QueryRequest
from pipeline.schemas import QueryResponse

logger = logging.getLogger(__name__)


class BenchmarkError(Exception):
    """Custom exception for benchmark errors."""
    pass


async def run_benchmark(
    queries_file: Path = Path("evaluation/queries.jsonl"),
    max_queries: int = 10,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run benchmark on a set of queries.
    
    Args:
        queries_file: Path to JSONL file with queries
        max_queries: Maximum number of queries to run
        
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
            logger.info(f"Processing query {i}/{len(queries)}: {query_data.get('query', '')[:50]}...")
            
            request = QueryRequest(
                query=query_data.get("query", ""),
                language=query_data.get("language", "en"),
            )
            
            query_start = time.perf_counter()
            
            try:
                response = await process_query(request)
                
                query_latency = (time.perf_counter() - query_start) * 1000
                
                result = {
                    "query_id": query_data.get("id", i),
                    "query": query_data.get("query", ""),
                    "answer": response.answer,
                    "expected_answer": query_data.get("answer", ""),
                    "confidence": response.confidence,
                    "latency_ms": response.latency_ms,
                    "evidence_count": len(response.evidence),
                    "passed": _check_answer(response.answer, query_data.get("answer", "")),
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Query {i} failed: {e}")
                results.append({
                    "query_id": query_data.get("id", i),
                    "query": query_data.get("query", ""),
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


def _check_answer(answer: str, expected: str) -> bool:
    """Check if answer matches expected answer."""
    if not expected:
        return True
    
    answer_lower = answer.lower().strip()
    expected_lower = expected.lower().strip()
    
    return expected_lower in answer_lower


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
