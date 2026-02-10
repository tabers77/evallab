"""Evaluation pipeline: Adapter -> Scorers -> ScoreVector.

Orchestrates the full evaluation flow from raw traces to scored results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_eval.core.models import Episode
from agent_eval.core.score import Issue, ScoreDimension, ScoreVector
from agent_eval.scorers.rules.deduction import RuleBasedScorer


@dataclass
class EvalResult:
    """Full evaluation result for a single episode."""

    episode: Episode
    score_vector: ScoreVector
    grade: str
    summary: str


class EvalPipeline:
    """Orchestrate evaluation: Adapter -> Scorers -> ScoreVector.

    Parameters
    ----------
    adapter
        A TraceAdapter to convert raw traces into Episodes.
    scorers
        List of Scorer instances to run on each Episode.
    rule_scorer
        Optional RuleBasedScorer for computing the overall 0-100 score.
        If not provided, a default one is created.
    """

    def __init__(
        self,
        adapter,
        scorers: list | None = None,
        rule_scorer: RuleBasedScorer | None = None,
    ) -> None:
        self.adapter = adapter
        self.scorers = scorers or []
        self.rule_scorer = rule_scorer or RuleBasedScorer()

    def evaluate(self, episode: Episode) -> EvalResult:
        """Evaluate a single Episode through all scorers."""
        all_dimensions: list[ScoreDimension] = []
        all_issues: list[Issue] = []

        for scorer in self.scorers:
            all_dimensions.extend(scorer.score(episode))
            all_issues.extend(scorer.detect_issues(episode))

        # Use RuleBasedScorer with the collected issues
        rule_dims = self.rule_scorer.score_with_issues(episode, all_issues)
        all_dimensions.extend(rule_dims)

        score_vector = ScoreVector(
            episode_id=episode.episode_id,
            dimensions=all_dimensions,
            issues=all_issues,
        )

        overall_dim = score_vector.dimension_by_name("overall_score")
        overall_value = overall_dim.value if overall_dim else 0.0
        grade = self.rule_scorer.get_grade(overall_value)

        summary = _generate_summary(episode, all_issues, overall_value)

        return EvalResult(
            episode=episode,
            score_vector=score_vector,
            grade=grade,
            summary=summary,
        )

    def evaluate_from_source(self, source: str, **kwargs) -> EvalResult:
        """Load an episode from source and evaluate it."""
        episode = self.adapter.load_episode(source, **kwargs)
        return self.evaluate(episode)

    def evaluate_batch(self, source: str, **kwargs) -> list[EvalResult]:
        """Load all episodes from source and evaluate each."""
        episodes = self.adapter.load_episodes(source, **kwargs)
        return [self.evaluate(ep) for ep in episodes]


def _generate_summary(
    episode: Episode,
    issues: list[Issue],
    score: float,
) -> str:
    """Generate a human-readable summary."""
    from agent_eval.core.score import Severity
    from agent_eval.core.models import StepKind

    lines: list[str] = []

    if score >= 90:
        lines.append("Excellent performance - no major issues detected.")
    elif score >= 70:
        lines.append("Good performance with some areas for improvement.")
    elif score >= 50:
        lines.append("Moderate performance - multiple issues detected.")
    else:
        lines.append("Poor performance - significant issues found.")

    tool_steps = episode.steps_by_kind(StepKind.TOOL_CALL)
    unique_tools = {s.tool_name for s in tool_steps if s.tool_name}
    successful = sum(1 for s in tool_steps if s.tool_succeeded is not False)
    llm_steps = episode.steps_by_kind(StepKind.LLM_CALL)

    lines.append(f"\nKey Metrics:")
    lines.append(f"  Agents Active: {len(episode.agents)}")
    lines.append(f"  Total Tool Calls: {len(tool_steps)} ({len(unique_tools)} unique)")
    if tool_steps:
        rate = successful / len(tool_steps) * 100
        lines.append(f"  Tool Success Rate: {rate:.0f}%")
    lines.append(f"  LLM Calls: {len(llm_steps)}")
    if episode.duration_seconds:
        lines.append(f"  Execution Time: {episode.duration_seconds:.0f}s")
    lines.append(f"  Final Answer: {'Present' if episode.final_answer else 'Missing'}")

    if issues:
        critical = sum(1 for i in issues if i.severity == Severity.CRITICAL)
        errors = sum(1 for i in issues if i.severity == Severity.ERROR)
        warnings = sum(1 for i in issues if i.severity == Severity.WARNING)
        lines.append(f"\nIssues Found: {len(issues)} total")
        if critical:
            lines.append(f"  CRITICAL: {critical}")
        if errors:
            lines.append(f"  ERROR: {errors}")
        if warnings:
            lines.append(f"  WARNING: {warnings}")
    else:
        lines.append("\nNo issues detected!")

    return "\n".join(lines)
