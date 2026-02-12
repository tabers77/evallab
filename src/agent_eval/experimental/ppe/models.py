"""Data models for PPE (Preference Proxy Evaluations) benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_eval.core.score import ScoreVector


@dataclass
class PreferencePair:
    """A pair of ScoreVectors with a ground-truth preference label.

    The ``preferred`` vector is the one that should receive a higher
    reward from a well-calibrated reward function.
    """

    preferred: ScoreVector
    rejected: ScoreVector
    domain: str = "default"

    def __post_init__(self) -> None:
        if self.preferred.episode_id == self.rejected.episode_id:
            raise ValueError(
                "PreferencePair must contain two distinct episodes "
                f"(got same id: {self.preferred.episode_id!r})"
            )


@dataclass
class BenchmarkSample:
    """K candidate ScoreVectors with ground-truth quality scores.

    Used for best-of-K selection and rank correlation metrics.
    """

    score_vectors: list[ScoreVector]
    ground_truth_scores: list[float]
    domain: str = "default"

    def __post_init__(self) -> None:
        if len(self.score_vectors) != len(self.ground_truth_scores):
            raise ValueError(
                f"score_vectors length ({len(self.score_vectors)}) must match "
                f"ground_truth_scores length ({len(self.ground_truth_scores)})"
            )
        if len(self.score_vectors) < 2:
            raise ValueError("BenchmarkSample requires at least 2 candidates")


@dataclass
class BenchmarkDataset:
    """Collection of preference pairs and/or benchmark samples."""

    name: str
    pairs: list[PreferencePair] = field(default_factory=list)
    samples: list[BenchmarkSample] = field(default_factory=list)

    @property
    def domains(self) -> list[str]:
        """Unique domain tags across all data."""
        seen: set[str] = set()
        for p in self.pairs:
            seen.add(p.domain)
        for s in self.samples:
            seen.add(s.domain)
        return sorted(seen)

    def __post_init__(self) -> None:
        if not self.pairs and not self.samples:
            raise ValueError(
                "BenchmarkDataset must contain at least one pair or sample"
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_pairs": len(self.pairs),
            "n_samples": len(self.samples),
            "domains": self.domains,
        }


@dataclass
class MetricResult:
    """Result of a single PPE metric evaluation."""

    metric_name: str
    value: float
    per_domain: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "value": round(self.value, 4),
            "per_domain": {k: round(v, 4) for k, v in self.per_domain.items()},
            "n_samples": self.n_samples,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResult:
    """Full benchmark results for one reward function."""

    dataset_name: str
    reward_fn_name: str
    metrics: list[MetricResult] = field(default_factory=list)

    def metric_by_name(self, name: str) -> MetricResult | None:
        for m in self.metrics:
            if m.metric_name == name:
                return m
        return None

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "reward_fn_name": self.reward_fn_name,
            "metrics": [m.to_dict() for m in self.metrics],
        }
