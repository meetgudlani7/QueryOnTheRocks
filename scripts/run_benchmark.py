"""
Benchmark Runner Script

Runs the RAG benchmark (evaluation/queries.jsonl) and the voice latency
benchmark against the live pipeline, then prints a report of measured
numbers only — no placeholders, no estimates.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Running this file directly (`python scripts/run_benchmark.py`) puts this
# file's own directory on sys.path, not the project root — without this,
# `from config import ...` below fails with ModuleNotFoundError. Mirrors
# scripts/build_index.py's identical bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import configure_logging
from evaluation.benchmark import run_benchmark, run_voice_latency_benchmark
from evaluation.metrics import calculate_metrics, calculate_percentiles

logger = logging.getLogger(__name__)

# Groq's on-demand tier caps llama-3.1-8b-instant at 6000 tokens/minute. A
# first unpaced run measured ~1067 tokens/request on average (see this
# project's benchmark log) — 6000/1067 ~= 5.6 req/min, i.e. one request per
# ~10.7s at the limit. 15s leaves headroom for above-average requests
# without inflating the total run past what the rate limit already demands.
REQUEST_DELAY_S = 15.0


async def main() -> int:
    configure_logging("INFO")

    logger.info("Running RAG benchmark (paced to stay under Groq's rate limit)...")
    results, summary = await run_benchmark(max_queries=200, request_delay_s=REQUEST_DELAY_S)
    metrics = calculate_metrics(results, include_per_query=False)

    logger.info("Running voice latency benchmark (30 STT + full-pipeline round trips)...")
    voice_results = await run_voice_latency_benchmark(n=30, request_delay_s=REQUEST_DELAY_S)
    stt_latencies = [r["stt_latency_ms"] for r in voice_results if "stt_latency_ms" in r]
    total_latencies = [r["total_latency_ms"] for r in voice_results if "total_latency_ms" in r]
    voice_failures = sum(1 for r in voice_results if "error" in r)

    print("\n" + "=" * 60)
    print("RAG BENCHMARK")
    print("=" * 60)
    print(f"Queries: {summary['total_queries']}  Passed: {summary['passed_queries']}  Success rate: {summary['success_rate']:.1f}%")
    print(f"RAG latency percentiles (ms): {metrics.get('latency_percentiles')}")
    print(f"Quality metrics: {metrics.get('quality_metrics')}")

    print("\n" + "=" * 60)
    print("VOICE LATENCY BENCHMARK")
    print("=" * 60)
    print(f"Runs: {len(voice_results)}  Failures: {voice_failures}")
    print(f"STT latency percentiles (ms): {calculate_percentiles(stt_latencies)}")
    print(f"Total voice latency percentiles (ms): {calculate_percentiles(total_latencies)}")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
