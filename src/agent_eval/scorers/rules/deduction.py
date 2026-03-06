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
      - Bonuses: for good answer, tool diversity, zero failures (configurable)
      - Clamped to [0, 100]

    Parameters
    ----------
    critical_weight
        Points deducted per CRITICAL issue.  Default ``25``.
    error_weight
        Points deducted per ERROR issue.  Default ``10``.
    warning_weight
        Points deducted per WARNING issue.  Default ``5``.
    answer_length_bonus
        Bonus points awarded when the final answer exceeds
        ``answer_length_threshold`` characters.  Default ``5``.
    answer_length_threshold
        Minimum final-answer length to earn the bonus.  Default ``500``.
    tool_diversity_bonus
        Bonus points for using ``tool_diversity_min`` or more unique
        tools.  Default ``3``.
    tool_diversity_min
        Minimum number of unique tools to earn the diversity bonus.
        Default ``3``.
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
        answer_length_bonus: float = 5,
        answer_length_threshold: int = 500,
        tool_diversity_bonus: float = 3,
        tool_diversity_min: int = 3,
        zero_failure_bonus: float = 2,
        grade_thresholds: dict[int, str] | None = None,
    ) -> None:
        self.critical_weight = critical_weight
        self.error_weight = error_weight
        self.warning_weight = warning_weight
        self.answer_length_bonus = answer_length_bonus
        self.answer_length_threshold = answer_length_threshold
        self.tool_diversity_bonus = tool_diversity_bonus
        self.tool_diversity_min = tool_diversity_min
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

        # Bonuses
        if (
            episode.final_answer
            and len(episode.final_answer) > self.answer_length_threshold
        ):
            score += self.answer_length_bonus

        tool_steps = episode.steps_by_kind(StepKind.TOOL_CALL)
        unique_tools = {s.tool_name for s in tool_steps if s.tool_name}
        if len(unique_tools) >= self.tool_diversity_min:
            score += self.tool_diversity_bonus

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
