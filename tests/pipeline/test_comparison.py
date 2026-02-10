"""Tests for pipeline.comparison — multi-run comparison."""

from agent_eval.core.models import Episode
from agent_eval.core.score import Issue, ScoreDimension, ScoreVector, Severity
from agent_eval.pipeline.comparison import (
    ComparisonResult,
    DimensionDelta,
    compare_batch,
    compare_results,
)
from agent_eval.pipeline.runner import EvalResult


def _make_result(
    score: float,
    dims: dict[str, float] | None = None,
    n_issues: int = 0,
) -> EvalResult:
    dimensions = [ScoreDimension(name="overall_score", value=score, max_value=100.0)]
    if dims:
        for name, val in dims.items():
            dimensions.append(ScoreDimension(name=name, value=val, max_value=1.0))

    issues = [Issue(Severity.WARNING, "T", f"I{i}") for i in range(n_issues)]
    ep = Episode(episode_id="test", steps=[], source_framework="test")
    sv = ScoreVector(episode_id="test", dimensions=dimensions, issues=issues)
    return EvalResult(episode=ep, score_vector=sv, grade="B", summary="")


class TestDimensionDelta:
    def test_improved(self):
        d = DimensionDelta(name="x", run_a_value=0.5, run_b_value=0.8)
        assert d.improved is True
        assert d.regressed is False
        assert abs(d.delta - 0.3) < 1e-9

    def test_regressed(self):
        d = DimensionDelta(name="x", run_a_value=0.9, run_b_value=0.6)
        assert d.regressed is True
        assert d.improved is False


class TestCompareResults:
    def test_basic_comparison(self):
        r_a = _make_result(80.0, dims={"accuracy": 0.8}, n_issues=3)
        r_b = _make_result(90.0, dims={"accuracy": 0.95}, n_issues=1)

        comp = compare_results(r_a, r_b, label_a="Old", label_b="New")

        assert comp.run_a_label == "Old"
        assert comp.run_b_label == "New"
        assert comp.score_delta == 10.0
        assert comp.improved is True
        assert comp.run_a_issue_count == 3
        assert comp.run_b_issue_count == 1

    def test_regression_detection(self):
        r_a = _make_result(90.0, dims={"accuracy": 0.95})
        r_b = _make_result(70.0, dims={"accuracy": 0.6})

        comp = compare_results(r_a, r_b)
        assert comp.improved is False
        assert len(comp.regressions) > 0

    def test_to_dict(self):
        r_a = _make_result(80.0)
        r_b = _make_result(85.0)
        comp = compare_results(r_a, r_b)
        d = comp.to_dict()
        assert "score_delta" in d
        assert d["improved"] is True


class TestCompareBatch:
    def test_batch_comparison(self):
        batch_a = [_make_result(80.0, n_issues=2), _make_result(70.0, n_issues=3)]
        batch_b = [_make_result(90.0, n_issues=1), _make_result(85.0, n_issues=1)]

        result = compare_batch(batch_a, batch_b, "v1", "v2")

        assert result["avg_score_a"] == 75.0
        assert result["avg_score_b"] == 87.5
        assert result["score_delta"] == 12.5
        assert result["total_issues_a"] == 5
        assert result["total_issues_b"] == 2
