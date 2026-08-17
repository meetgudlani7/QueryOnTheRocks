"""
Guardrail Threshold Calibration Script (roadmap Phase 18)

MIN_RETRIEVAL_SCORE and MIN_GROUNDEDNESS_SIMILARITY (config/settings.py)
were shipped as documented placeholders — "provisional default pending
real tuning against an evaluation set." This script does that tuning: it
sweeps both thresholds against evaluation/queries.jsonl and reports which
combination actually maximizes answer quality, using only measured
numbers, never guesses.

Mirrors scripts/run_benchmark.py's pacing: Groq's on-demand tier caps
llama-3.1-8b-instant at 6000 tokens/minute, so every query in the sweep
is paced by the same measured REQUEST_DELAY_S. A full grid over N
combinations costs N times a single benchmark run's wall-clock time, so
this defaults to a small, stratified, representative query sample rather
than the full 157-query set — enough to see a real signal without an
hours-long run. Pass --max-queries-per-type for a more thorough (slower)
sweep.
"""

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Running this file directly (`python scripts/calibrate_guardrails.py`) puts
# this file's own directory on sys.path, not the project root — without
# this, `from config import ...` below fails with ModuleNotFoundError.
# Mirrors scripts/build_index.py's identical bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import configure_logging, settings
from evaluation.benchmark import run_benchmark
from evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)

# Same measured basis as scripts/run_benchmark.py's REQUEST_DELAY_S.
DEFAULT_REQUEST_DELAY_S = 15.0

DEFAULT_RETRIEVAL_SCORES = [0.05, 0.15, 0.30]
DEFAULT_GROUNDEDNESS_SIMILARITIES = [0.20, 0.35, 0.50]


def _stratified_sample(queries_file: Path, max_per_type: int) -> List[Dict[str, Any]]:
    """
    Takes up to max_per_type queries from each `type` category (factoid,
    semantic, unanswerable, off_topic, injection) instead of the first N
    lines of the file — the file is not shuffled by category, so a plain
    head(N) would badly over-represent whatever type happens to lead the
    file and never exercise the refusal categories at all, which are
    exactly what MIN_RETRIEVAL_SCORE is meant to calibrate.
    """
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_type[record.get("type", "unknown")].append(record)

    sample: List[Dict[str, Any]] = []
    for query_type, records in sorted(by_type.items()):
        sample.extend(records[:max_per_type])

    logger.info(
        f"Stratified sample: {len(sample)} queries across {len(by_type)} types "
        f"(up to {max_per_type} each) from {queries_file}"
    )
    return sample


def _combined_score(quality: Dict[str, Any]) -> float:
    """
    Refusal accuracy and grounded-answer rate are weighted equally, per
    the roadmap's explicit reasoning: a system that answers confidently
    and wrongly is worse than one that refuses too often, so neither
    metric should be allowed to dominate the other. Missing metrics
    (None, e.g. no answerable/refusal queries in a given sample) score as
    0 rather than being excluded, so an incomplete sample can't produce a
    misleadingly high combined score.
    """
    refusal_accuracy = quality.get("refusal_accuracy") or 0.0
    grounded_answer_rate = quality.get("grounded_answer_rate") or 0.0
    return 0.5 * refusal_accuracy + 0.5 * grounded_answer_rate


async def _run_one_combination(
    sample_file: Path,
    max_queries: int,
    request_delay_s: float,
    min_retrieval_score: float,
    min_groundedness_similarity: float,
) -> Dict[str, Any]:
    # Mutating the live settings singleton is deliberate and safe here:
    # pipeline/guardrails.py reads settings.MIN_RETRIEVAL_SCORE and
    # settings.MIN_GROUNDEDNESS_SIMILARITY fresh on every call (no
    # caching), and this script is the only thing running in this
    # process, so there's no concurrent request whose guardrails would be
    # affected by a threshold change mid-run.
    settings.MIN_RETRIEVAL_SCORE = min_retrieval_score
    settings.MIN_GROUNDEDNESS_SIMILARITY = min_groundedness_similarity

    logger.info(
        f"Running combination: MIN_RETRIEVAL_SCORE={min_retrieval_score}, "
        f"MIN_GROUNDEDNESS_SIMILARITY={min_groundedness_similarity}"
    )
    results, summary = await run_benchmark(
        queries_file=sample_file, max_queries=max_queries, request_delay_s=request_delay_s
    )
    metrics = calculate_metrics(results, include_per_query=False)
    quality = metrics.get("quality_metrics", {})

    row = {
        "min_retrieval_score": min_retrieval_score,
        "min_groundedness_similarity": min_groundedness_similarity,
        "refusal_accuracy": quality.get("refusal_accuracy"),
        "grounded_answer_rate": quality.get("grounded_answer_rate"),
        "average_answer_f1": quality.get("average_answer_f1"),
        "json_validity_rate": quality.get("json_validity_rate"),
        "avg_latency_ms": metrics.get("avg_latency_ms"),
        "combined_score": _combined_score(quality),
    }
    logger.info(f"Result: {row}")
    return row


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-file", type=Path, default=Path("evaluation/queries.jsonl"))
    parser.add_argument(
        "--max-queries-per-type", type=int, default=2,
        help="Queries sampled per category, per threshold combination (default: 2 — keep this small, it multiplies total runtime)",
    )
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY_S)
    parser.add_argument(
        "--retrieval-scores", type=str, default=",".join(str(v) for v in DEFAULT_RETRIEVAL_SCORES),
        help="Comma-separated MIN_RETRIEVAL_SCORE values to sweep",
    )
    parser.add_argument(
        "--groundedness-similarities", type=str, default=",".join(str(v) for v in DEFAULT_GROUNDEDNESS_SIMILARITIES),
        help="Comma-separated MIN_GROUNDEDNESS_SIMILARITY values to sweep",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/guardrail_calibration_results.json"),
        help="Where to save the full sweep results for audit/record-keeping",
    )
    args = parser.parse_args()

    configure_logging("INFO")

    retrieval_scores = [float(v) for v in args.retrieval_scores.split(",")]
    groundedness_similarities = [float(v) for v in args.groundedness_similarities.split(",")]
    total_combinations = len(retrieval_scores) * len(groundedness_similarities)

    sample = _stratified_sample(args.queries_file, args.max_queries_per_type)
    if not sample:
        logger.error(f"No queries found in {args.queries_file}")
        return 1

    estimated_minutes = (total_combinations * len(sample) * args.request_delay) / 60
    logger.info(
        f"Sweeping {len(retrieval_scores)} x {len(groundedness_similarities)} = "
        f"{total_combinations} combinations over {len(sample)} queries each "
        f"({total_combinations * len(sample)} total pipeline calls, "
        f"~{estimated_minutes:.0f} min at {args.request_delay}s/query pacing)"
    )

    # Original values, restored at the end regardless of outcome — this
    # script must never leave the process (or a copy of this file re-run
    # against a live API) in a different default state than it started.
    original_retrieval_score = settings.MIN_RETRIEVAL_SCORE
    original_groundedness_similarity = settings.MIN_GROUNDEDNESS_SIMILARITY

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for record in sample:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        sample_file = Path(f.name)

    rows: List[Dict[str, Any]] = []
    try:
        for min_retrieval_score in retrieval_scores:
            for min_groundedness_similarity in groundedness_similarities:
                row = await _run_one_combination(
                    sample_file, len(sample), args.request_delay,
                    min_retrieval_score, min_groundedness_similarity,
                )
                rows.append(row)
    finally:
        settings.MIN_RETRIEVAL_SCORE = original_retrieval_score
        settings.MIN_GROUNDEDNESS_SIMILARITY = original_groundedness_similarity
        sample_file.unlink(missing_ok=True)

    rows.sort(key=lambda r: r["combined_score"], reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_size_per_combination": len(sample),
                "queries_file": str(args.queries_file),
                "current_defaults": {
                    "MIN_RETRIEVAL_SCORE": original_retrieval_score,
                    "MIN_GROUNDEDNESS_SIMILARITY": original_groundedness_similarity,
                },
                "results_sorted_best_first": rows,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("GUARDRAIL THRESHOLD CALIBRATION — results sorted best first")
    print("=" * 72)
    header = f"{'retrieval':>10} {'grounded':>10} {'refusal_acc':>12} {'grounded_rate':>14} {'avg_f1':>8} {'combined':>10}"
    print(header)
    for r in rows:
        print(
            f"{r['min_retrieval_score']:>10} {r['min_groundedness_similarity']:>10} "
            f"{_fmt(r['refusal_accuracy']):>12} {_fmt(r['grounded_answer_rate']):>14} "
            f"{_fmt(r['average_answer_f1']):>8} {_fmt(r['combined_score']):>10}"
        )

    best = rows[0]
    print(
        f"\nRecommended: MIN_RETRIEVAL_SCORE={best['min_retrieval_score']}, "
        f"MIN_GROUNDEDNESS_SIMILARITY={best['min_groundedness_similarity']} "
        f"(combined_score={_fmt(best['combined_score'])})"
    )
    print(
        f"Current .env defaults: MIN_RETRIEVAL_SCORE={original_retrieval_score}, "
        f"MIN_GROUNDEDNESS_SIMILARITY={original_groundedness_similarity}"
    )
    print(f"\nFull results saved to {args.output}")
    print(
        "\nNote: this ran on a small stratified sample for wall-clock feasibility "
        "(Groq rate-limit pacing). Re-run with a higher --max-queries-per-type before "
        "treating the recommendation as final."
    )

    return 0


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
