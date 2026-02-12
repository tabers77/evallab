"""PPE (Preference Proxy Evaluations) — reward model benchmarking.

Experimental module adapting the PPE methodology from arXiv:2410.14872
to benchmark evallab's RewardFunction implementations.
"""

from .models import (
    BenchmarkDataset,
    BenchmarkResult,
    BenchmarkSample,
    MetricResult,
    PreferencePair,
)
from .metrics import (
    ALL_METRICS,
    ALL_PAIR_METRICS,
    ALL_SAMPLE_METRICS,
    best_of_k,
    brier_score,
    kendall_tau,
    pairwise_accuracy,
    separability,
    spearman_correlation,
)
from .runner import BenchmarkRunner
from .synthetic import SyntheticDatasetBuilder, perturb_score_vector
from .report import (
    benchmark_to_dict,
    benchmark_to_text,
    comparison_to_dict,
    comparison_to_text,
)

__all__ = [
    # Models
    "PreferencePair",
    "BenchmarkSample",
    "BenchmarkDataset",
    "MetricResult",
    "BenchmarkResult",
    # Metrics
    "pairwise_accuracy",
    "best_of_k",
    "spearman_correlation",
    "kendall_tau",
    "separability",
    "brier_score",
    "ALL_METRICS",
    "ALL_PAIR_METRICS",
    "ALL_SAMPLE_METRICS",
    # Runner
    "BenchmarkRunner",
    # Synthetic
    "SyntheticDatasetBuilder",
    "perturb_score_vector",
    # Report
    "benchmark_to_text",
    "comparison_to_text",
    "benchmark_to_dict",
    "comparison_to_dict",
]
