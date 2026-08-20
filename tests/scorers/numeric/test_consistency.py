"""Tests for scorers.numeric.consistency — NumericConsistencyScorer."""

from datetime import datetime, timezone

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Severity
from agent_eval.scorers.numeric.consistency import NumericConsistencyScorer


def _make_episode(
    final_answer: str,
    tool_results: list[dict],
) -> Episode:
    """Helper to build an Episode with tool call steps."""
    steps = []
    for tr in tool_results:
        steps.append(
            Step(
                kind=StepKind.TOOL_CALL,
                agent_id="",
                agent_name="",
                tool_name=tr.get("tool", "test_tool"),
                tool_result=tr.get("result"),
                tool_succeeded=True,
                metadata={
                    "raw_event": {
                        "type": "ToolCall",
                        "tool": tr.get("tool", "test_tool"),
                        "result": tr.get("result"),
                    }
                },
            )
        )
    return Episode(
        episode_id="test",
        steps=steps,
        source_framework="test",
        final_answer=final_answer,
    )


class TestNumericConsistencyScorer:
    def test_name(self):
        scorer = NumericConsistencyScorer()
        assert scorer.name == "numeric_consistency"

    def test_no_fabrication(self):
        ep = _make_episode(
            final_answer="Revenue is 283,399,382.94",
            tool_results=[
                {"tool": "get_finances", "result": {"REVENUE": 283399382.94}}
            ],
        )
        scorer = NumericConsistencyScorer()
        dims = scorer.score(ep)
        assert dims[0].name == "numeric_accuracy"
        assert dims[0].value == 1.0

        issues = scorer.detect_issues(ep)
        assert len(issues) == 0

    def test_fabrication_detected(self):
        ep = _make_episode(
            final_answer="Revenue is 350M",
            tool_results=[
                {"tool": "get_finances", "result": {"REVENUE": 283399382.94}}
            ],
        )
        scorer = NumericConsistencyScorer()
        issues = scorer.detect_issues(ep)
        assert len(issues) > 0
        assert issues[0].severity == Severity.CRITICAL
        assert issues[0].category == "Data Fabrication"

    def test_score_reflects_fabrications(self):
        ep = _make_episode(
            final_answer="Revenue is 350M with volume of 15,000",
            tool_results=[
                {
                    "tool": "get_finances",
                    "result": {"REVENUE": 283399382.94, "VOLUME": 15000},
                }
            ],
        )
        scorer = NumericConsistencyScorer()
        dims = scorer.score(ep)
        # 15000 matches but 350M doesn't -> 1 match, 1 fabrication
        assert dims[0].value < 1.0

    def test_no_answer_numbers(self):
        ep = _make_episode(
            final_answer="No numbers in this answer",
            tool_results=[
                {"tool": "get_finances", "result": {"REVENUE": 283399382.94}}
            ],
        )
        scorer = NumericConsistencyScorer()
        dims = scorer.score(ep)
        assert dims[0].value == 1.0  # No numbers to fabricate

    def test_no_tool_numbers(self):
        ep = _make_episode(
            final_answer="Revenue is 350M",
            tool_results=[],
        )
        scorer = NumericConsistencyScorer()
        issues = scorer.detect_issues(ep)
        assert len(issues) == 0  # Can't validate without tool numbers

    def test_tolerance(self):
        # 5% tolerance: 283M vs 283399382 should match
        ep = _make_episode(
            final_answer="Revenue is 283M",
            tool_results=[
                {"tool": "get_finances", "result": {"REVENUE": 283399382.94}}
            ],
        )
        scorer = NumericConsistencyScorer(tolerance=0.05)
        # 283000000 vs 283399382 = 0.14% error -> within tolerance
        issues = scorer.detect_issues(ep)
        assert len(issues) == 0

    def test_small_numbers_skipped(self):
        ep = _make_episode(
            final_answer="Growth rate is 0.5",
            tool_results=[{"tool": "get_data", "result": {"growth": 0.8}}],
        )
        scorer = NumericConsistencyScorer()
        issues = scorer.detect_issues(ep)
        # Numbers < 1 are skipped
        assert len(issues) == 0

    def test_approximate_numbers_not_flagged(self):
        """Numbers preceded by ~ should not be flagged as fabrications."""
        ep = _make_episode(
            final_answer="Total spend is 17,322.77 with potential savings of ~953",
            tool_results=[
                {"tool": "get_spend", "result": {"total": 17322.77}}
            ],
        )
        scorer = NumericConsistencyScorer()
        issues = scorer.detect_issues(ep)
        # 17,322.77 matches tool; ~953 is approximate so not flagged
        assert len(issues) == 0

    def test_derived_keywords_not_flagged(self):
        """Numbers in derived context (reduce by, save, target) are skipped."""
        ep = _make_episode(
            final_answer=(
                "Total spend is 17,322.77. "
                "You could target a reduce by 50% to save 8,661"
            ),
            tool_results=[
                {"tool": "get_spend", "result": {"total": 17322.77}}
            ],
        )
        scorer = NumericConsistencyScorer()
        issues = scorer.detect_issues(ep)
        # 17,322.77 matches; 50 and 8,661 are derived
        assert len(issues) == 0

    def test_percent_range_not_flagged(self):
        """Numbers inside percent range patterns are skipped."""
        ep = _make_episode(
            final_answer=(
                "Total is 94,004.24. Savings could be 5.5% to 7.5%"
            ),
            tool_results=[
                {"tool": "get_spend", "result": {"total": 94004.24}}
            ],
        )
        scorer = NumericConsistencyScorer()
        issues = scorer.detect_issues(ep)
        assert len(issues) == 0

    def test_scenario_table_not_flagged(self):
        """Scenario table with approximate and derived numbers is not flagged."""
        ep = _make_episode(
            final_answer=(
                "Total spend: 17,322.77\n"
                "| Scenario | Target savings |\n"
                "| Defensive | reduce by 50%, save approximately 953 |\n"
                "| Target | reduce by 100%, target savings ~2,790 |\n"
            ),
            tool_results=[
                {"tool": "get_spend", "result": {"total": 17322.77}}
            ],
        )
        scorer = NumericConsistencyScorer()
        issues = scorer.detect_issues(ep)
        # Only 17,322.77 is factual and matches tool data
        assert len(issues) == 0

    def test_factual_fabrication_still_detected(self):
        """A non-derived number that doesn't match tools is still flagged."""
        ep = _make_episode(
            final_answer="Total spend is 25,000.00",
            tool_results=[
                {"tool": "get_spend", "result": {"total": 17322.77}}
            ],
        )
        scorer = NumericConsistencyScorer()
        issues = scorer.detect_issues(ep)
        assert len(issues) == 1
        assert issues[0].category == "Data Fabrication"
