"""
Benchmark History Tracker (roadmap Phase 24)

Runs the RAG benchmark and appends one summary row to
evaluation/benchmark_history.jsonl — a simple, append-only,
git-trackable log of quality metrics over time, so a regression (from a
threshold change, a model swap, a corpus update) shows up as a diff in
that file instead of only being discoverable by a user noticing worse
answers. Intended to run on a schedule (see
.github/workflows/nightly-benchmark.yml) or manually before/after a
change that might affect answer quality.
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Running this file directly (`python scripts/track_benchmark.py`) puts
# this file's own directory on sys.path, not the project root — without
# this, `from config import ...` below fails with ModuleNotFoundError.
# Mirrors scripts/build_index.py's identical bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import configure_logging
from evaluation.benchmark import run_benchmark
from evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_FILE = Path("evaluation/benchmark_history.jsonl")
# Same measured basis as scripts/run_benchmark.py's REQUEST_DELAY_S — a
# full 157-query run at this pacing takes ~40 minutes, hence this being a
# scheduled/manual tool rather than something run on every commit.
DEFAULT_REQUEST_DELAY_S = 15.0
# A metric moving by less than this is treated as noise, not a real
# regression or improvement — chosen to match the granularity already
# used for calibration decisions elsewhere (scripts/calibrate_guardrails.py).
REGRESSION_THRESHOLD = 0.05

TRACKED_METRICS = (
    "recall_at_5", "evidence_hit_rate", "grounded_answer_rate",
    "refusal_accuracy", "json_validity_rate", "average_answer_f1",
)


def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping corrupt history line in {path}")
    return rows


def _append_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compare(previous: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Lines describing any tracked metric that moved by >= REGRESSION_THRESHOLD, best-effort labeled improved/regressed."""
    lines = []
    for key in TRACKED_METRICS:
        prev_val, cur_val = previous.get(key), current.get(key)
        if prev_val is None or cur_val is None:
            continue
        delta = cur_val - prev_val
        if abs(delta) >= REGRESSION_THRESHOLD:
            direction = "IMPROVED" if delta > 0 else "REGRESSED"
            lines.append(f"{direction}: {key} {prev_val:.3f} -> {cur_val:.3f} ({delta:+.3f})")
    return lines


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-file", type=Path, default=Path("evaluation/queries.jsonl"))
    parser.add_argument("--max-queries", type=int, default=157)
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY_S)
    parser.add_argument("--history-file", type=Path, default=DEFAULT_HISTORY_FILE)
    parser.add_argument(
        "--fail-on-regression", action="store_true",
        help=f"Exit non-zero if any tracked quality metric drops by >= {REGRESSION_THRESHOLD} vs. the previous run",
    )
    args = parser.parse_args()

    configure_logging("INFO")

    history = _load_history(args.history_file)
    previous: Optional[Dict[str, Any]] = history[-1]["quality_metrics"] if history else None

    logger.info(f"Running benchmark ({args.max_queries} queries max, {args.request_delay}s pacing)...")
    results, summary = await run_benchmark(
        queries_file=args.queries_file, max_queries=args.max_queries, request_delay_s=args.request_delay
    )
    metrics = calculate_metrics(results, include_per_query=False)
    quality = metrics.get("quality_metrics", {})

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_sha(),
        "total_queries": summary["total_queries"],
        "success_rate": summary["success_rate"],
        "avg_latency_ms": metrics.get("avg_latency_ms"),
        "latency_percentiles": metrics.get("latency_percentiles"),
        "quality_metrics": quality,
    }
    _append_row(args.history_file, row)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULT")
    print("=" * 60)
    print(f"Commit: {row['git_commit']}  Queries: {row['total_queries']}  Success rate: {row['success_rate']:.1f}%")
    print(f"Quality metrics: {quality}")

    regressions: List[str] = []
    if previous:
        comparisons = _compare(previous, quality)
        if comparisons:
            print("\nChanges vs. previous run:")
            for line in comparisons:
                print(f"  {line}")
            regressions = [line for line in comparisons if line.startswith("REGRESSED")]
        else:
            print(f"\nNo tracked metric moved >= {REGRESSION_THRESHOLD} vs. the previous run.")
    else:
        print("\n(First tracked run — nothing to compare against yet.)")

    print(f"\nAppended to {args.history_file} ({len(history) + 1} runs tracked)")

    if args.fail_on_regression and regressions:
        logger.error(f"{len(regressions)} metric(s) regressed by >= {REGRESSION_THRESHOLD} — failing per --fail-on-regression")
        return 1
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
