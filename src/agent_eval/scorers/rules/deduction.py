"""Rule-based deduction scorer.

Ported from log_evaluator.py:_calculate_score — base-100 scoring with
severity-based deductions and bonus points.
"""

from __future__ import annotations

from agent_eval.core.models import Episode, StepKind
from agent_eval.core.score import Issue, ScoreDimension, Severity


class RuleBasedScorer:
    """Deduction-based scorer that produces a 0-100 score.

    Scoring formula:
      - Base: 100
      - Deductions: per CRITICAL / ERROR / WARNING (configurable)
      - Bonus: zero tool failures (configurable)
      - Clamped to [0, 100]

    Parameters
    ----------
    critical_weight
        Points deducted per CRITICAL issue.  Default ``25``.
    error_weight
        Points deducted per ERROR issue.  Default ``10``.
    warning_weight
        Points deducted per WARNING issue.  Default ``5``.
    zero_failure_bonus
        Bonus points when all tool calls succeeded.  Default ``2``.
    grade_thresholds
        Dict mapping minimum scores to letter grades, evaluated in
        descending order.  Default ``{90: "A", 80: "B", 70: "C", 60: "D"}``.
        Any score below the lowest threshold receives ``"F"``.
    """

    def __init__(
        self,
        critical_weight: float = 25,
        error_weight: float = 10,
        warning_weight: float = 5,
        zero_failure_bonus: float = 2,
        grade_thresholds: dict[int, str] | None = None,
    ) -> None:
        self.critical_weight = critical_weight
        self.error_weight = error_weight
        self.warning_weight = warning_weight
        self.zero_failure_bonus = zero_failure_bonus
        self.grade_thresholds = grade_thresholds or {
            90: "A",
            80: "B",
            70: "C",
            60: "D",
        }

    @property
    def name(self) -> str:
        return "rule_based"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        """Compute score using self-detected issues (no external issues)."""
        return self.score_with_issues(episode, [])

    def detect_issues(self, episode: Episode) -> list[Issue]:
        """This scorer does not detect issues; delegate to IssueDetectorScorer."""
        return []

    def score_with_issues(
        self,
        episode: Episode,
        issues: list[Issue],
    ) -> list[ScoreDimension]:
        """Compute score given pre-detected issues.

        This is the primary entry point when used inside a pipeline
        where issues come from a separate IssueDetectorScorer.
        """
        score = 100.0

        for issue in issues:
            if issue.severity == Severity.CRITICAL:
                score -= self.critical_weight
            elif issue.severity == Severity.ERROR:
                score -= self.error_weight
            elif issue.severity == Severity.WARNING:
                score -= self.warning_weight

        # Bonuses.
        #
        # 0.3.0 removed the answer-length and tool-diversity bonuses (audit
        # F4). They paid for verbosity and busywork: measured on this scorer,
        # padding an answer from 55 to 1,376 characters moved it 92.0 -> 97.0,
        # and going from 1 tool to 5 moved it 92.0 -> 95.0, with no change in
        # substance either time. An agent optimised against that metric learns
        # to pad and to spray tool calls.
        #
        # They were invisible with zero issues, because the base is 100 and the
        # score clamps to [0, 100] — which is why this went unnoticed. See
        # tests/scorers/rules/test_reward_hacking_probe.py, which holds a fixed
        # non-empty issue set precisely so the bonuses are observable.
        #
        # zero_failure_bonus is kept deliberately: "no tool call failed" cannot
        # be farmed by padding, so it is not a reward-hacking surface.
        tool_steps = episode.steps_by_kind(StepKind.TOOL_CALL)
        if tool_steps and all(s.tool_succeeded is not False for s in tool_steps):
            score += self.zero_failure_bonus

        score = max(0.0, min(100.0, score))

        return [
            ScoreDimension(
                name="overall_score",
                value=round(score, 1),
                max_value=100.0,
                source=self.name,
            )
        ]

    def get_grade(self, score: float) -> str:
        """Convert a 0-100 score to a letter grade."""
        for threshold in sorted(self.grade_thresholds, reverse=True):
            if score >= threshold:
                return self.grade_thresholds[threshold]
        return "F"
