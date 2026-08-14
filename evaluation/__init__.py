"""
Evaluation Module

Handles benchmarking and evaluation of the RAG system.
"""

from .benchmark import run_benchmark
from .metrics import calculate_metrics

__all__ = ["run_benchmark", "calculate_metrics"]
