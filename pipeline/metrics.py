"""
Metrics Collection Module

Tracks performance metrics for each pipeline stage.
"""

import time
import logging
from typing import Dict, Optional
from collections import defaultdict
from threading import Lock

from .schemas import ProcessingStage

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and reports pipeline performance metrics.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance
    
    def _init(self):
        self._metrics: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "total_latency_ms": 0.0,
            "min_latency_ms": float("inf"),
            "max_latency_ms": 0.0,
        })
        self._start_time = time.time()
    
    @classmethod
    def record(cls, stage: ProcessingStage, latency_ms: float):
        """
        Record a metric for a pipeline stage.
        
        Args:
            stage: Processing stage
            latency_ms: Latency in milliseconds
        """
        instance = cls()
        stage_name = stage.value
        
        metrics = instance._metrics[stage_name]
        metrics["count"] += 1
        metrics["total_latency_ms"] += latency_ms
        metrics["min_latency_ms"] = min(metrics["min_latency_ms"], latency_ms)
        metrics["max_latency_ms"] = max(metrics["max_latency_ms"], latency_ms)
        
        logger.debug(f"Recorded {stage_name}: {latency_ms:.2f}ms")
    
    @classmethod
    def get_metrics(cls) -> Dict:
        """
        Get all collected metrics.
        
        Returns:
            Dictionary with metrics for all stages
        """
        instance = cls()
        result = {}
        
        for stage_name, metrics in instance._metrics.items():
            if metrics["count"] > 0:
                result[stage_name] = {
                    "count": metrics["count"],
                    "avg_latency_ms": metrics["total_latency_ms"] / metrics["count"],
                    "min_latency_ms": metrics["min_latency_ms"],
                    "max_latency_ms": metrics["max_latency_ms"],
                    "total_latency_ms": metrics["total_latency_ms"],
                }
        
        return result
    
    @classmethod
    def get_summary(cls) -> Dict:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        instance = cls()
        metrics = cls.get_metrics()

        total_requests = sum(m["count"] for m in metrics.values())
        total_latency = sum(m["total_latency_ms"] for m in metrics.values())

        return {
            "total_requests": total_requests,
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / max(total_requests, 1),
            "stages": metrics,
            "uptime_seconds": time.time() - instance._start_time,
        }
    
    @classmethod
    def reset(cls):
        """Reset all collected metrics."""
        instance = cls()
        instance._metrics.clear()
        instance._start_time = time.time()
        logger.info("Metrics reset")
