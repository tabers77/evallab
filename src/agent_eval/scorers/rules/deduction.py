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
      - Deductions: -25 per CRITICAL, -10 per ERROR, -5 per WARNING
      - Bonuses: +5 for good answer, +3 for tool diversity, +2 for zero failures
      - Clamped to [0, 100]

    Parameters
    ----------
    issue_scorer
        Optional scorer to use for issue detection.  When ``None``,
        issues must be passed via the ``issues`` parameter in
        :meth:`score_with_issues`.
    """

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
                score -= 25
            elif issue.severity == Severity.ERROR:
                score -= 10
            elif issue.severity == Severity.WARNING:
                score -= 5

        # Bonuses
        if episode.final_answer and len(episode.final_answer) > 500:
            score += 5

        tool_steps = episode.steps_by_kind(StepKind.TOOL_CALL)
        unique_tools = {s.tool_name for s in tool_steps if s.tool_name}
        if len(unique_tools) >= 3:
            score += 3

        if tool_steps and all(s.tool_succeeded is not False for s in tool_steps):
            score += 2

        score = max(0.0, min(100.0, score))

        return [
            ScoreDimension(
                name="overall_score",
                value=round(score, 1),
                max_value=100.0,
                source=self.name,
            )
        ]

    @staticmethod
    def get_grade(score: float) -> str:
        """Convert a 0-100 score to a letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        return "F"
