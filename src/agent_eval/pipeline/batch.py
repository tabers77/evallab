"""Batch evaluation utilities.

Extends the core EvalPipeline with batch-oriented helpers for
evaluating directories of logs and collecting aggregate statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from agent_eval.core.score import Severity
from agent_eval.pipeline.runner import EvalPipeline, EvalResult

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Aggregate results from evaluating multiple episodes."""

    results: list[EvalResult] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def scores(self) -> list[float]:
        """Overall scores from all results."""
        out: list[float] = []
        for r in self.results:
            dim = r.score_vector.dimension_by_name("overall_score")
            out.append(dim.value if dim else 0.0)
        return out

    @property
    def avg_score(self) -> float:
        s = self.scores
        return sum(s) / len(s) if s else 0.0

    @property
    def best_score(self) -> float:
        s = self.scores
        return max(s) if s else 0.0

    @property
    def worst_score(self) -> float:
        s = self.scores
        return min(s) if s else 0.0

    @property
    def grade_distribution(self) -> dict[str, int]:
        grades: dict[str, int] = {}
        for r in self.results:
            grades[r.grade] = grades.get(r.grade, 0) + 1
        return grades

    @property
    def total_issues(self) -> int:
        return sum(len(r.score_vector.issues) for r in self.results)

    @property
    def issue_severity_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            for issue in r.score_vector.issues:
                key = issue.severity.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "avg_score": round(self.avg_score, 1),
            "best_score": round(self.best_score, 1),
            "worst_score": round(self.worst_score, 1),
            "grade_distribution": self.grade_distribution,
            "total_issues": self.total_issues,
            "issue_severity_counts": self.issue_severity_counts,
        }


def evaluate_batch(pipeline: EvalPipeline, source: str) -> BatchResult:
    """Evaluate all episodes from a source directory.

    Parameters
    ----------
    pipeline
        Configured EvalPipeline instance.
    source
        Path to a directory containing log files.

    Returns
    -------
    BatchResult with individual and aggregate results.
    """
    results = pipeline.evaluate_batch(source)
    return BatchResult(results=results)


def evaluate_paths(pipeline: EvalPipeline, paths: list[str]) -> BatchResult:
    """Evaluate specific log paths.

    Parameters
    ----------
    pipeline
        Configured EvalPipeline instance.
    paths
        List of paths to individual log files or directories.
    """
    results: list[EvalResult] = []
    for path in paths:
        try:
            result = pipeline.evaluate_from_source(path)
            results.append(result)
        except Exception as e:
            logger.warning("Failed to evaluate %s: %s", path, e)
    return BatchResult(results=results)
