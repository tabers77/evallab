"""Benchmark runner for PPE reward model evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import BenchmarkDataset, BenchmarkResult, MetricResult
from .metrics import ALL_PAIR_METRICS, ALL_SAMPLE_METRICS, ALL_METRICS


@dataclass
class BenchmarkRunner:
    """Orchestrates PPE benchmark evaluation of reward functions.

    Parameters
    ----------
    dataset
        The benchmark dataset to evaluate against.
    metric_names
        Which metrics to compute.  *None* means all applicable metrics.
    separability_margin
        Margin parameter passed to the separability metric.
    """

    dataset: BenchmarkDataset
    metric_names: list[str] | None = None
    separability_margin: float = 0.1

    def _resolve_metrics(self) -> list[str]:
        """Determine which metrics to run based on available data."""
        if self.metric_names is not None:
            return [m for m in self.metric_names if m in ALL_METRICS]

        names: list[str] = []
        if self.dataset.pairs:
            names.extend(ALL_PAIR_METRICS.keys())
        if self.dataset.samples:
            names.extend(ALL_SAMPLE_METRICS.keys())
        return names

    def run(self, reward_fn: object, label: str = "") -> BenchmarkResult:
        """Evaluate a single reward function against the dataset.

        Parameters
        ----------
        reward_fn
            Any object with a ``compute(ScoreVector) -> float`` method,
            matching the ``RewardFunction`` protocol.
        label
            Human-readable name for the reward function.
        """
        fn_name = label or getattr(reward_fn, "name", None) or type(reward_fn).__name__
        compute = reward_fn.compute
        metric_names = self._resolve_metrics()

        results: list[MetricResult] = []
        for name in metric_names:
            metric_fn = ALL_METRICS[name]
            if name == "separability":
                result = metric_fn(self.dataset, compute, margin=self.separability_margin)
            else:
                result = metric_fn(self.dataset, compute)
            results.append(result)

        return BenchmarkResult(
            dataset_name=self.dataset.name,
            reward_fn_name=fn_name,
            metrics=results,
        )

    def run_comparison(
        self,
        reward_fns: list[tuple[object, str]],
    ) -> list[BenchmarkResult]:
        """Evaluate multiple reward functions for side-by-side comparison.

        Parameters
        ----------
        reward_fns
            List of ``(reward_fn, label)`` tuples.
        """
        return [self.run(fn, label) for fn, label in reward_fns]
