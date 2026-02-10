"""Tests for scorers.rules.deduction — RuleBasedScorer."""

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Issue, Severity
from agent_eval.scorers.rules.deduction import RuleBasedScorer


def _make_episode(**kwargs) -> Episode:
    defaults = {
        "episode_id": "test",
        "steps": [],
        "source_framework": "test",
    }
    defaults.update(kwargs)
    return Episode(**defaults)


class TestRuleBasedScorer:
    def test_name(self):
        scorer = RuleBasedScorer()
        assert scorer.name == "rule_based"

    def test_perfect_score_no_issues(self):
        ep = _make_episode(
            final_answer="A" * 600,
            steps=[
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name="t1",
                    tool_succeeded=True,
                ),
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name="t2",
                    tool_succeeded=True,
                ),
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name="t3",
                    tool_succeeded=True,
                ),
            ],
        )
        scorer = RuleBasedScorer()
        dims = scorer.score_with_issues(ep, [])
        assert (
            dims[0].value == 100.0
        )  # 100 + 5 (answer) + 3 (tools) + 2 (no failures) capped

    def test_critical_deduction(self):
        ep = _make_episode()
        issues = [
            Issue(Severity.CRITICAL, "Test", "Critical 1"),
            Issue(Severity.CRITICAL, "Test", "Critical 2"),
        ]
        scorer = RuleBasedScorer()
        dims = scorer.score_with_issues(ep, issues)
        assert dims[0].value == 50.0  # 100 - 25 - 25

    def test_mixed_deductions(self):
        ep = _make_episode()
        issues = [
            Issue(Severity.CRITICAL, "Test", "Critical"),
            Issue(Severity.ERROR, "Test", "Error"),
            Issue(Severity.WARNING, "Test", "Warning"),
        ]
        scorer = RuleBasedScorer()
        dims = scorer.score_with_issues(ep, issues)
        assert dims[0].value == 60.0  # 100 - 25 - 10 - 5

    def test_score_clamped_to_zero(self):
        ep = _make_episode()
        issues = [Issue(Severity.CRITICAL, "Test", f"C{i}") for i in range(10)]
        scorer = RuleBasedScorer()
        dims = scorer.score_with_issues(ep, issues)
        assert dims[0].value == 0.0

    def test_bonus_for_good_answer(self):
        ep = _make_episode(final_answer="A" * 600)
        scorer = RuleBasedScorer()
        dims = scorer.score_with_issues(ep, [])
        assert dims[0].value >= 100.0  # 100 + 5 bonus, capped at 100

    def test_bonus_for_tool_diversity(self):
        ep = _make_episode(
            steps=[
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name=f"tool_{i}",
                    tool_succeeded=True,
                )
                for i in range(4)
            ],
        )
        scorer = RuleBasedScorer()
        dims = scorer.score_with_issues(ep, [])
        # 100 + 3 (diversity) + 2 (no failures) = 105, capped at 100
        assert dims[0].value == 100.0

    def test_dimension_metadata(self):
        ep = _make_episode()
        scorer = RuleBasedScorer()
        dims = scorer.score_with_issues(ep, [])
        assert dims[0].name == "overall_score"
        assert dims[0].max_value == 100.0
        assert dims[0].source == "rule_based"


class TestGetGrade:
    def test_grade_a(self):
        assert RuleBasedScorer.get_grade(95) == "A"
        assert RuleBasedScorer.get_grade(90) == "A"

    def test_grade_b(self):
        assert RuleBasedScorer.get_grade(85) == "B"
        assert RuleBasedScorer.get_grade(80) == "B"

    def test_grade_c(self):
        assert RuleBasedScorer.get_grade(75) == "C"
        assert RuleBasedScorer.get_grade(70) == "C"

    def test_grade_d(self):
        assert RuleBasedScorer.get_grade(65) == "D"
        assert RuleBasedScorer.get_grade(60) == "D"

    def test_grade_f(self):
        assert RuleBasedScorer.get_grade(55) == "F"
        assert RuleBasedScorer.get_grade(0) == "F"
