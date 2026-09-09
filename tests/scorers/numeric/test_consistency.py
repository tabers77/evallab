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


class TestEvidenceIncompleteAbstains:
    """An episode may declare its own tool evidence incomplete.

    A conversation turn that summarises documents fetched in an *earlier* turn
    carries figures this episode cannot trace. Observed in production: turn 3
    of a 3-turn chat scored 0.0 with ~40 CRITICAL "Data Fabrication" issues,
    every one of them a correct figure from turn 2 — the closest match was
    always the single number in turn 3's own tool call.
    """

    def _episode(self, incomplete: bool):
        from agent_eval.core.models import Episode

        meta = {
            "raw_events": [
                {"type": "ToolCall", "tool": "t", "arguments": {}, "result": "868.62"}
            ]
        }
        if incomplete:
            meta["evidence_incomplete"] = True
        return Episode(
            episode_id="x",
            steps=[],
            source_framework="t",
            final_answer="Kraftliner at 795.09 and Testliner at 762.70",
            metadata=meta,
        )

    def test_issues_are_withheld(self):
        scorer = NumericConsistencyScorer()
        assert scorer.detect_issues(self._episode(True)) == []

    def test_the_same_episode_reports_without_the_flag(self):
        """Guard: the flag must be what silences it, not a broken detector."""
        scorer = NumericConsistencyScorer()
        assert len(scorer.detect_issues(self._episode(False))) > 0

    def test_dimension_abstains_rather_than_scoring_zero(self):
        """Absence of measurement must not read as a bad measurement."""
        dims = NumericConsistencyScorer().score(self._episode(True))
        assert len(dims) == 1
        assert dims[0].abstained is True

    def test_dimension_is_scored_without_the_flag(self):
        dims = NumericConsistencyScorer().score(self._episode(False))
        assert dims[0].abstained is False


class TestGradeSpecCollisionStillCaught:
    """A fabricated figure colliding with a grade spec must still be flagged.

    The serious half of intelligence-platform GAP-046. Admitting paperboard
    grade specs to the reference pool adds spurious *matches*, not only
    spurious misses, and in this domain they land squarely inside the
    plausible band for the metric being checked: observed EBITDA EUR/t in the
    captured answer spans -14 to 243, while the specs injected by the grade
    names alone are 80, 115, 260, 330, 370, 475, 480 and 665.

    So a fabricated "EUR 262/t" matched "CKB 260 mN" within tolerance=0.05 and
    passed silently on 0.3.7. Verified against the shipped 0.3.7 wheel while
    diagnosing this: the episode below produced no issues at all.

    Tolerance cannot fix this, because the contamination is on the reference
    side rather than the answer side, and min_value cannot either, because the
    specs sit inside the legitimate range of the metric.
    """

    TOOL_RESULT = "grade=CKB 260 mN | EBITDA_EUR=23127619.15 | TONS=181632.14"

    def _episode(self, answer: str) -> Episode:
        return _make_episode(
            final_answer=answer,
            tool_results=[{"tool": "profitability", "result": self.TOOL_RESULT}],
        )

    def test_a_figure_colliding_with_a_grade_spec_is_reported(self):
        issues = NumericConsistencyScorer(tolerance=0.05).detect_issues(
            self._episode("The margin was EUR 262/t.")
        )
        assert len(issues) == 1
        assert issues[0].severity == Severity.CRITICAL
        assert "262" in issues[0].description

    def test_the_spec_is_not_offered_as_the_closest_match(self):
        """The pool was cleaned, so no "closest:" can point at a stiffness.

        Asserted positively -- the nearest surviving pool member is the tonnage
        -- rather than as "260 is absent", which a different closest value
        merely containing those digits would satisfy by accident.
        """
        issues = NumericConsistencyScorer(tolerance=0.05).detect_issues(
            self._episode("The margin was EUR 262/t.")
        )
        assert "181,632.14" in issues[0].description
        assert "260" not in issues[0].description

    def test_a_sourced_figure_is_still_not_flagged(self):
        """Guard: the pool must be cleaned, not emptied into all-fabricated."""
        issues = NumericConsistencyScorer(tolerance=0.05).detect_issues(
            self._episode("EBITDA was 23127619.15 EUR on 181632.14 tonnes.")
        )
        assert issues == []

    def test_the_grade_name_in_the_answer_is_not_itself_a_fabrication(self):
        """The answer names the grade too; masking both sides keeps it silent."""
        issues = NumericConsistencyScorer(tolerance=0.05).detect_issues(
            self._episode("CKB 260 mN delivered 23127619.15 EUR.")
        )
        assert issues == []
