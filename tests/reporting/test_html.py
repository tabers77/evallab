"""Tests for reporting.html — HTML report output."""

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Issue, ScoreDimension, ScoreVector, Severity
from agent_eval.pipeline.runner import EvalResult
from agent_eval.reporting.html import format_html_report, format_html_batch


def _make_result(
    score: float = 85.0,
    grade: str = "B",
    n_issues: int = 0,
    has_answer: bool = True,
) -> EvalResult:
    steps = [
        Step(kind=StepKind.MESSAGE, agent_id="u", agent_name="user", content="Hello"),
        Step(
            kind=StepKind.TOOL_CALL,
            agent_id="a",
            agent_name="agent",
            tool_name="search",
            tool_succeeded=True,
        ),
        Step(kind=StepKind.LLM_CALL, agent_id="a", agent_name="agent", model="gpt-4o"),
        Step(
            kind=StepKind.MESSAGE, agent_id="a", agent_name="agent", content="Response"
        ),
    ]
    dims = [
        ScoreDimension(
            name="overall_score", value=score, max_value=100.0, source="rule_based"
        ),
        ScoreDimension(name="quality", value=0.8, max_value=1.0, source="llm_judge"),
    ]
    issues = [Issue(Severity.WARNING, "Test", f"Issue {i}") for i in range(n_issues)]
    ep = Episode(
        episode_id="test-ep",
        steps=steps,
        source_framework="test",
        final_answer="The answer" if has_answer else None,
        metadata={"source_path": "/test/log.txt"},
    )
    sv = ScoreVector(episode_id="test-ep", dimensions=dims, issues=issues)
    return EvalResult(
        episode=ep, score_vector=sv, grade=grade, summary="Good performance."
    )


class TestFormatHtmlReport:
    def test_contains_html_structure(self):
        result = _make_result()
        html = format_html_report(result)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<body>" in html

    def test_contains_score(self):
        result = _make_result(score=92.0, grade="A")
        html = format_html_report(result)
        assert "92/100" in html
        assert "Grade: A" in html

    def test_contains_framework(self):
        result = _make_result()
        html = format_html_report(result)
        assert "test" in html

    def test_contains_metrics_when_verbose(self):
        result = _make_result()
        html = format_html_report(result, verbose=True)
        assert "Metrics" in html
        assert "Tool Calls" in html
        assert "LLM Calls" in html

    def test_no_metrics_when_not_verbose(self):
        result = _make_result()
        html = format_html_report(result, verbose=False)
        assert "LLM Calls" not in html

    def test_contains_dimensions(self):
        result = _make_result()
        html = format_html_report(result, verbose=True)
        assert "overall_score" in html
        assert "quality" in html
        assert "Score Dimensions" in html

    def test_contains_issues(self):
        result = _make_result(n_issues=3)
        html = format_html_report(result)
        assert "Issues (3)" in html
        assert "WARNING" in html
        assert "Issue 0" in html

    def test_no_issues_message(self):
        result = _make_result(n_issues=0)
        html = format_html_report(result)
        assert "No issues detected" in html

    def test_html_escaping(self):
        """Ensure HTML special characters are escaped."""
        ep = Episode(
            episode_id="test",
            steps=[],
            source_framework="test",
            metadata={"source_path": "<script>alert('xss')</script>"},
        )
        sv = ScoreVector(
            episode_id="test",
            dimensions=[
                ScoreDimension(name="overall_score", value=50.0, max_value=100.0)
            ],
        )
        result = EvalResult(
            episode=ep, score_vector=sv, grade="F", summary="<b>bad</b>"
        )
        html = format_html_report(result)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_issue_severity_colors(self):
        dims = [ScoreDimension(name="overall_score", value=40.0, max_value=100.0)]
        issues = [
            Issue(Severity.CRITICAL, "Errors", "Critical error"),
            Issue(Severity.ERROR, "Errors", "Regular error"),
            Issue(Severity.WARNING, "Quality", "A warning"),
        ]
        ep = Episode(episode_id="t", steps=[], source_framework="test")
        sv = ScoreVector(episode_id="t", dimensions=dims, issues=issues)
        result = EvalResult(episode=ep, score_vector=sv, grade="F", summary="Bad")
        html = format_html_report(result)
        assert "Critical" in html
        assert "Error" in html
        assert "Warning" in html


class TestFormatHtmlBatch:
    def test_empty_results(self):
        html = format_html_batch([])
        assert "No results" in html

    def test_single_result(self):
        result = _make_result()
        html = format_html_batch([result])
        assert "Batch Evaluation Report" in html
        assert "Total runs: 1" in html

    def test_multiple_results(self):
        results = [
            _make_result(score=90.0, grade="A"),
            _make_result(score=70.0, grade="C"),
            _make_result(score=80.0, grade="B"),
        ]
        html = format_html_batch(results)
        assert "Total runs: 3" in html
        assert "80.0/100" in html  # average

    def test_results_sorted_by_score(self):
        results = [
            _make_result(score=60.0, grade="D"),
            _make_result(score=90.0, grade="A"),
        ]
        html = format_html_batch(results)
        # Grade A should appear before Grade D in sorted table
        a_pos = html.find("Grade A") if "Grade A" in html else html.find(">A<")
        d_pos = html.find("Grade D") if "Grade D" in html else html.find(">D<")
        # The A-graded result should appear first (lower position)
        assert a_pos < d_pos
