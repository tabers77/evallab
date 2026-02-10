"""Tests for scorers.rules.issue_detector — IssueDetectorScorer."""

from datetime import datetime, timezone

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Severity
from agent_eval.scorers.rules.issue_detector import IssueDetectorScorer


def _make_episode(**kwargs) -> Episode:
    defaults = {
        "episode_id": "test",
        "steps": [],
        "source_framework": "test",
        "final_answer": "A detailed answer with enough content to pass length checks. "
        * 5,
        "metadata": {},
    }
    defaults.update(kwargs)
    return Episode(**defaults)


class TestIssueDetectorScorerBasics:
    def test_name(self):
        scorer = IssueDetectorScorer()
        assert scorer.name == "issue_detector"

    def test_perfect_episode_score(self, multi_agent_episode):
        scorer = IssueDetectorScorer()
        dims = scorer.score(multi_agent_episode)
        assert dims[0].name == "issue_free"
        assert dims[0].value > 0


class TestAnswerQuality:
    def test_missing_final_answer(self):
        ep = _make_episode(final_answer=None)
        scorer = IssueDetectorScorer()
        issues = scorer.detect_issues(ep)
        critical = [i for i in issues if i.severity == Severity.CRITICAL]
        assert any("final answer" in i.description.lower() for i in critical)

    def test_short_final_answer(self):
        ep = _make_episode(final_answer="Short")
        scorer = IssueDetectorScorer()
        issues = scorer.detect_issues(ep)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any("short" in i.description.lower() for i in warnings)


class TestToolUsage:
    def test_no_tools_called(self):
        ep = _make_episode(
            steps=[
                Step(
                    kind=StepKind.MESSAGE,
                    agent_id="a",
                    agent_name="Agent1",
                    content="msg",
                ),
            ]
        )
        scorer = IssueDetectorScorer()
        issues = scorer.detect_issues(ep)
        critical = [i for i in issues if i.severity == Severity.CRITICAL]
        assert any("no tools" in i.description.lower() for i in critical)

    def test_limited_tool_diversity(self):
        ep = _make_episode(
            steps=[
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name="same_tool",
                    tool_succeeded=True,
                ),
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name="same_tool",
                    tool_succeeded=True,
                ),
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name="same_tool",
                    tool_succeeded=True,
                ),
            ]
        )
        scorer = IssueDetectorScorer()
        issues = scorer.detect_issues(ep)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any(
            "limited" in i.description.lower() or "diversity" in i.description.lower()
            for i in warnings
        )

    def test_high_failure_rate(self):
        ep = _make_episode(
            steps=[
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name="t1",
                    tool_succeeded=False,
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
                    tool_succeeded=False,
                ),
            ]
        )
        scorer = IssueDetectorScorer()
        issues = scorer.detect_issues(ep)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert any("failure rate" in i.description.lower() for i in errors)


class TestAgentCoordination:
    def test_single_agent_warning(self):
        ep = _make_episode(
            steps=[
                Step(
                    kind=StepKind.MESSAGE,
                    agent_id="a1",
                    agent_name="Agent1",
                    content="Only me here",
                ),
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name="tool1",
                    tool_succeeded=True,
                ),
            ]
        )
        scorer = IssueDetectorScorer()
        issues = scorer.detect_issues(ep)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any(
            "one agent" in i.description.lower()
            or "delegation" in i.description.lower()
            for i in warnings
        )


class TestEfficiency:
    def test_long_execution_time(self):
        ep = _make_episode(
            started_at=datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 1, 15, 14, 10, 0, tzinfo=timezone.utc),
            steps=[
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name="t1",
                    tool_succeeded=True,
                ),
            ],
        )
        scorer = IssueDetectorScorer(max_execution_seconds=300)
        issues = scorer.detect_issues(ep)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any("execution time" in i.description.lower() for i in warnings)

    def test_excessive_llm_calls(self):
        llm_steps = [
            Step(kind=StepKind.LLM_CALL, agent_id="", agent_name="", model="gpt-4o")
            for _ in range(35)
        ]
        llm_steps.append(
            Step(
                kind=StepKind.TOOL_CALL,
                agent_id="",
                agent_name="",
                tool_name="t1",
                tool_succeeded=True,
            ),
        )
        ep = _make_episode(steps=llm_steps)
        scorer = IssueDetectorScorer(max_llm_calls=30)
        issues = scorer.detect_issues(ep)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any("llm calls" in i.description.lower() for i in warnings)


class TestDataAccuracy:
    def test_fact_check_failures(self):
        ep = _make_episode(
            steps=[
                Step(
                    kind=StepKind.FACT_CHECK,
                    agent_id="FE",
                    agent_name="FinanceExpert",
                    metadata={"verdict": "FAIL", "reasoning": ["Error"]},
                ),
                Step(
                    kind=StepKind.FACT_CHECK,
                    agent_id="FE",
                    agent_name="FinanceExpert",
                    metadata={"verdict": "PASS", "reasoning": ["OK"]},
                ),
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name="t1",
                    tool_succeeded=True,
                ),
            ]
        )
        scorer = IssueDetectorScorer()
        issues = scorer.detect_issues(ep)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert any("fact checking" in i.description.lower() for i in errors)

    def test_error_detection_from_raw_content(self):
        ep = _make_episode(
            metadata={
                "raw_content": "Exception occurred: ValueError\nTraceback (most recent call last):"
            },
            steps=[
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name="t1",
                    tool_succeeded=True,
                ),
            ],
        )
        scorer = IssueDetectorScorer()
        issues = scorer.detect_issues(ep)
        critical = [i for i in issues if i.severity == Severity.CRITICAL]
        assert any(
            "exception" in i.description.lower() or "traceback" in i.description.lower()
            for i in critical
        )
