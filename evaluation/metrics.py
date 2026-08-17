"""
Metrics Calculation Module

Calculates evaluation metrics for the RAG system.
"""

import re
from typing import List, Dict, Any, Optional
from collections import Counter
import statistics

# Query categories that expect a grounded answer vs. a refusal.
# Mirrors the "type" values in evaluation/queries.jsonl.
ANSWERABLE_TYPES = {"factoid", "semantic"}
REFUSAL_TYPES = {"unanswerable", "off_topic", "injection"}

# Below this token-overlap F1, two answer strings are considered not a
# match. 0.5 is the conventional SQuAD-style cutoff for "close enough" —
# raw F1 is still reported separately (see calculate_average_f1) for
# anyone who wants the continuous signal instead of a binary hit/miss.
F1_MATCH_THRESHOLD = 0.5


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — standard QA-eval normalization."""
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def calculate_f1(prediction: str, reference: str) -> float:
    """
    Token-overlap F1 between a prediction and a reference answer
    (SQuAD-style) — gives partial credit for paraphrases, unlike exact
    substring containment, which barely ever survives an LLM restating an
    answer in its own words.
    """
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _answer_hits(text: str, expected_answer: str) -> bool:
    """F1 >= F1_MATCH_THRESHOLD against the expected answer. Empty expected answer never matches."""
    if not expected_answer:
        return False
    return calculate_f1(text, expected_answer) >= F1_MATCH_THRESHOLD


def calculate_average_f1(results: List[Dict[str, Any]]) -> Optional[float]:
    """Mean token-overlap F1 between generated and expected answers, over answerable queries."""
    scored = [
        r for r in results
        if r.get("type") in ANSWERABLE_TYPES and r.get("expected_answer") and isinstance(r.get("answer"), str)
    ]
    if not scored:
        return None
    return statistics.mean(calculate_f1(r["answer"], r["expected_answer"]) for r in scored)


def calculate_recall_at_k(results: List[Dict[str, Any]], k: int = 5) -> Optional[float]:
    """
    Fraction of answerable queries where the relevant passage was retrieved
    among the first k results.

    Prefers exact document_id matching against "relevant_chunk_ids" (the
    real MSMARCO-XI is_selected relevance judgment for that query, resolved
    to actually-indexed chunk ids — see evaluation/queries.jsonl) when a
    query carries it; this is the textbook-correct way to measure
    retrieval recall, not a proxy. Falls back to F1 text matching against
    the expected answer for queries without a resolved relevant chunk
    (e.g. hand-authored queries not sourced from the indexed corpus).

    Returns None if there are no answerable queries with evidence to score.
    """
    scored = [r for r in results if r.get("type") in ANSWERABLE_TYPES and "evidence" in r]
    if not scored:
        return None
    hits = 0
    counted = 0
    for r in scored:
        relevant_ids = r.get("relevant_chunk_ids")
        if relevant_ids:
            counted += 1
            hits += 1 if set(r.get("evidence_ids", [])[:k]) & set(relevant_ids) else 0
        elif r.get("expected_answer"):
            counted += 1
            hits += 1 if any(_answer_hits(p, r["expected_answer"]) for p in r["evidence"][:k]) else 0
    return hits / counted if counted else None


def calculate_evidence_hit_rate(results: List[Dict[str, Any]]) -> Optional[float]:
    """
    Fraction of answerable queries where the relevant passage was retrieved
    anywhere (unrestricted position, unlike recall@K). Same ID-first,
    F1-fallback approach as calculate_recall_at_k.
    """
    scored = [r for r in results if r.get("type") in ANSWERABLE_TYPES and "evidence" in r]
    if not scored:
        return None
    hits = 0
    counted = 0
    for r in scored:
        relevant_ids = r.get("relevant_chunk_ids")
        if relevant_ids:
            counted += 1
            hits += 1 if set(r.get("evidence_ids", [])) & set(relevant_ids) else 0
        elif r.get("expected_answer"):
            counted += 1
            hits += 1 if any(_answer_hits(p, r["expected_answer"]) for p in r["evidence"]) else 0
    return hits / counted if counted else None


def calculate_grounded_answer_rate(results: List[Dict[str, Any]]) -> Optional[float]:
    """Fraction of answerable queries where the pipeline returned grounded=True."""
    scored = [r for r in results if r.get("type") in ANSWERABLE_TYPES and "grounded" in r]
    if not scored:
        return None
    return sum(1 for r in scored if r["grounded"]) / len(scored)


def calculate_refusal_accuracy(results: List[Dict[str, Any]]) -> Optional[float]:
    """
    Fraction of queries where the grounded/refused decision matched what the
    query's category expects: answerable types should end up grounded=True,
    refusal types (unanswerable/off-topic/injection) should end up
    grounded=False.
    """
    scored = [r for r in results if r.get("type") and "grounded" in r]
    if not scored:
        return None
    correct = 0
    for r in scored:
        expects_answer = r["type"] in ANSWERABLE_TYPES
        correct += 1 if (r["grounded"] == expects_answer) else 0
    return correct / len(scored)


def calculate_percentiles(values: List[float], percentiles: List[int] = [50, 70, 95, 100]) -> Dict[str, Optional[float]]:
    """Nearest-rank percentiles over measured values — no interpolation, no invented numbers."""
    if not values:
        return {f"p{p}": None for p in percentiles}
    ordered = sorted(values)
    n = len(ordered)
    result = {}
    for p in percentiles:
        rank = max(1, -(-p * n // 100))  # ceil(p/100 * n), at least 1
        result[f"p{p}"] = ordered[min(rank, n) - 1]
    return result


def calculate_json_validity_rate(results: List[Dict[str, Any]]) -> Optional[float]:
    """
    Fraction of queries that produced a schema-valid QueryResponse instead
    of erroring out. Malformed LLM JSON is retried/repaired inside
    pipeline.generation before ever reaching this layer, so this measures
    the deepest validity the evaluation harness can observe from the
    outside: did the pipeline return a well-formed structured response.
    """
    if not results:
        return None
    valid = sum(1 for r in results if "error" not in r and isinstance(r.get("answer"), str))
    return valid / len(results)


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
        metrics["latency_percentiles"] = calculate_percentiles(latencies)

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
    
    # Quality metrics (recall@K, evidence hit rate, grounded-answer rate,
    # refusal accuracy, JSON validity rate) — each is None if the results
    # don't carry the fields needed to compute it (e.g. older result shape).
    metrics["quality_metrics"] = {
        "recall_at_5": calculate_recall_at_k(results, k=5),
        "evidence_hit_rate": calculate_evidence_hit_rate(results),
        "grounded_answer_rate": calculate_grounded_answer_rate(results),
        "refusal_accuracy": calculate_refusal_accuracy(results),
        "json_validity_rate": calculate_json_validity_rate(results),
        "average_answer_f1": calculate_average_f1(results),
    }

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
