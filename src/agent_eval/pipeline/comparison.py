"""Multi-run comparison reports.

Compares evaluation results across multiple runs or configurations
to identify regressions and improvements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_eval.pipeline.runner import EvalResult


@dataclass
class DimensionDelta:
    """Change in a score dimension between two runs."""

    name: str
    run_a_value: float
    run_b_value: float

    @property
    def delta(self) -> float:
        return self.run_b_value - self.run_a_value

    @property
    def improved(self) -> bool:
        return self.delta > 0

    @property
    def regressed(self) -> bool:
        return self.delta < 0


@dataclass
class ComparisonResult:
    """Comparison between two evaluation results."""

    run_a_label: str
    run_b_label: str
    run_a_score: float
    run_b_score: float
    dimension_deltas: list[DimensionDelta] = field(default_factory=list)
    run_a_issue_count: int = 0
    run_b_issue_count: int = 0

    @property
    def score_delta(self) -> float:
        return self.run_b_score - self.run_a_score

    @property
    def improved(self) -> bool:
        return self.score_delta > 0

    @property
    def regressions(self) -> list[DimensionDelta]:
        return [d for d in self.dimension_deltas if d.regressed]

    @property
    def improvements(self) -> list[DimensionDelta]:
        return [d for d in self.dimension_deltas if d.improved]

    def to_dict(self) -> dict:
        return {
            "run_a": self.run_a_label,
            "run_b": self.run_b_label,
            "run_a_score": self.run_a_score,
            "run_b_score": self.run_b_score,
            "score_delta": round(self.score_delta, 2),
            "improved": self.improved,
            "regressions": [
                {"name": d.name, "delta": round(d.delta, 4)} for d in self.regressions
            ],
            "improvements": [
                {"name": d.name, "delta": round(d.delta, 4)} for d in self.improvements
            ],
            "run_a_issues": self.run_a_issue_count,
            "run_b_issues": self.run_b_issue_count,
        }


def compare_results(
    result_a: EvalResult,
    result_b: EvalResult,
    label_a: str = "Run A",
    label_b: str = "Run B",
) -> ComparisonResult:
    """Compare two EvalResults dimension-by-dimension.

    Parameters
    ----------
    result_a, result_b
        Two evaluation results to compare.
    label_a, label_b
        Human-readable labels for each run.
    """
    score_a = _get_overall(result_a)
    score_b = _get_overall(result_b)

    # Build dimension deltas
    dims_a = {d.name: d for d in result_a.score_vector.dimensions}
    dims_b = {d.name: d for d in result_b.score_vector.dimensions}

    all_names = sorted(set(dims_a.keys()) | set(dims_b.keys()))
    deltas: list[DimensionDelta] = []
    for name in all_names:
        val_a = dims_a[name].normalized if name in dims_a else 0.0
        val_b = dims_b[name].normalized if name in dims_b else 0.0
        deltas.append(DimensionDelta(name=name, run_a_value=val_a, run_b_value=val_b))

    return ComparisonResult(
        run_a_label=label_a,
        run_b_label=label_b,
        run_a_score=score_a,
        run_b_score=score_b,
        dimension_deltas=deltas,
        run_a_issue_count=len(result_a.score_vector.issues),
        run_b_issue_count=len(result_b.score_vector.issues),
    )


def compare_batch(
    results_a: list[EvalResult],
    results_b: list[EvalResult],
    label_a: str = "Baseline",
    label_b: str = "Experiment",
) -> dict:
    """Compare aggregate statistics between two batches.

    Returns a summary dict with average scores and deltas.
    """
    avg_a = _avg_score(results_a)
    avg_b = _avg_score(results_b)

    issues_a = sum(len(r.score_vector.issues) for r in results_a)
    issues_b = sum(len(r.score_vector.issues) for r in results_b)

    return {
        "label_a": label_a,
        "label_b": label_b,
        "count_a": len(results_a),
        "count_b": len(results_b),
        "avg_score_a": round(avg_a, 1),
        "avg_score_b": round(avg_b, 1),
        "score_delta": round(avg_b - avg_a, 1),
        "total_issues_a": issues_a,
        "total_issues_b": issues_b,
        "issue_delta": issues_b - issues_a,
    }


def _get_overall(result: EvalResult) -> float:
    dim = result.score_vector.dimension_by_name("overall_score")
    return dim.value if dim else 0.0


def _avg_score(results: list[EvalResult]) -> float:
    if not results:
        return 0.0
    return sum(_get_overall(r) for r in results) / len(results)
