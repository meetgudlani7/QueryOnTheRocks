"""
Metrics Calculation Module

Calculates evaluation metrics for the RAG system.
"""

from typing import List, Dict, Any, Optional
from collections import Counter
import statistics


def calculate_metrics(
    results: List[Dict[str, Any]],
    include_per_query: bool = True,
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics from benchmark results.
    
    Args:
        results: List of benchmark results
        include_per_query: Whether to include per-query metrics
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}
    
    # Basic metrics
    total = len(results)
    if total == 0:
        return {"error": "No results to calculate metrics"}
    
    # Accuracy metrics
    passed = sum(1 for r in results if r.get("passed", False))
    metrics["accuracy"] = passed / total
    metrics["passed_count"] = passed
    metrics["failed_count"] = total - passed
    
    # Latency metrics
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    if latencies:
        metrics["avg_latency_ms"] = statistics.mean(latencies)
        metrics["median_latency_ms"] = statistics.median(latencies)
        metrics["min_latency_ms"] = min(latencies)
        metrics["max_latency_ms"] = max(latencies)
        metrics["latency_std_dev"] = statistics.stdev(latencies) if len(latencies) > 1 else 0
    
    # Confidence metrics
    confidences = [r["confidence"] for r in results if "confidence" in r]
    if confidences:
        metrics["avg_confidence"] = statistics.mean(confidences)
        metrics["min_confidence"] = min(confidences)
        metrics["max_confidence"] = max(confidences)
    
    # Evidence metrics
    evidence_counts = [r["evidence_count"] for r in results if "evidence_count" in r]
    if evidence_counts:
        metrics["avg_evidence_count"] = statistics.mean(evidence_counts)
    
    # Target latency comparison
    target = 200.0  # 200ms target
    under_target = sum(1 for l in latencies if l <= target)
    metrics["under_target_pct"] = (under_target / len(latencies)) * 100 if latencies else 0
    metrics["target_latency_ms"] = target
    
    # Include per-query metrics if requested
    if include_per_query:
        metrics["per_query"] = results
    
    return metrics


def calculate_precision_recall(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate precision and recall metrics.
    
    Args:
        results: List of benchmark results
        
    Returns:
        Dictionary with precision and recall metrics
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for result in results:
        passed = result.get("passed", False)
        
        if passed:
            true_positives += 1
        else:
            if result.get("answer"):
                false_positives += 1
            else:
                false_negatives += 1
    
    total_predicted = true_positives + false_positives
    total_actual = true_positives + false_negatives
    
    precision = true_positives / total_predicted if total_predicted > 0 else 0
    recall = true_positives / total_actual if total_actual > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
