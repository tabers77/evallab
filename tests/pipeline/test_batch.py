"""Tests for pipeline.batch — BatchResult and batch evaluation."""

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Issue, ScoreDimension, ScoreVector, Severity
from agent_eval.pipeline.batch import BatchResult
from agent_eval.pipeline.runner import EvalResult


def _make_result(score: float, grade: str, n_issues: int = 0) -> EvalResult:
    issues = [Issue(Severity.WARNING, "Test", f"Issue {i}") for i in range(n_issues)]
    ep = Episode(episode_id="test", steps=[], source_framework="test")
    sv = ScoreVector(
        episode_id="test",
        dimensions=[ScoreDimension(name="overall_score", value=score, max_value=100.0)],
        issues=issues,
    )
    return EvalResult(episode=ep, score_vector=sv, grade=grade, summary="")


class TestBatchResult:
    def test_empty_batch(self):
        batch = BatchResult()
        assert batch.count == 0
        assert batch.avg_score == 0.0

    def test_single_result(self):
        batch = BatchResult(results=[_make_result(85.0, "B")])
        assert batch.count == 1
        assert batch.avg_score == 85.0
        assert batch.best_score == 85.0
        assert batch.worst_score == 85.0

    def test_multiple_results(self):
        batch = BatchResult(
            results=[
                _make_result(90.0, "A"),
                _make_result(70.0, "C"),
                _make_result(80.0, "B"),
            ]
        )
        assert batch.count == 3
        assert batch.avg_score == 80.0
        assert batch.best_score == 90.0
        assert batch.worst_score == 70.0

    def test_grade_distribution(self):
        batch = BatchResult(
            results=[
                _make_result(95.0, "A"),
                _make_result(92.0, "A"),
                _make_result(75.0, "C"),
            ]
        )
        grades = batch.grade_distribution
        assert grades["A"] == 2
        assert grades["C"] == 1

    def test_issue_counts(self):
        batch = BatchResult(
            results=[
                _make_result(90.0, "A", n_issues=0),
                _make_result(70.0, "C", n_issues=3),
                _make_result(50.0, "F", n_issues=5),
            ]
        )
        assert batch.total_issues == 8

    def test_to_dict(self):
        batch = BatchResult(results=[_make_result(85.0, "B", n_issues=2)])
        d = batch.to_dict()
        assert d["count"] == 1
        assert d["avg_score"] == 85.0
        assert d["total_issues"] == 2
