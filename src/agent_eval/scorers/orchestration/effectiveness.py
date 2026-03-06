"""Orchestration effectiveness scorer.

Evaluates the quality of multi-agent workflow orchestration
independent of individual agent reasoning quality. Based on the
distinction between in-context reasoning (orchestration) and
post-training reasoning from "Agentic Reasoning for Large Language
Models" (Wei et al., 2026).

Produces 5 dimensions: delegation_efficiency, tool_strategy,
coordination_overhead, recovery_effectiveness, and termination_quality.
"""

from __future__ import annotations

from collections import defaultdict

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Issue, ScoreDimension, Severity

_PRODUCTIVE_KINDS = {StepKind.TOOL_CALL, StepKind.TOOL_RESULT, StepKind.FACT_CHECK}


class OrchestrationScorer:
    """Scores the effectiveness of multi-agent orchestration.

    Analyzes Episode-level structure (step ordering, agent transitions,
    tool patterns) rather than message content.

    Parameters
    ----------
    min_productive_message_len
        Minimum character count for a MESSAGE step to be considered
        productive.  Default ``20``.
    overhead_threshold
        If the productive-step ratio falls below this value the scorer
        emits a "High coordination overhead" warning.  Default ``0.3``.
    """

    def __init__(
        self,
        min_productive_message_len: int = 20,
        overhead_threshold: float = 0.3,
    ) -> None:
        self.min_productive_message_len = min_productive_message_len
        self.overhead_threshold = overhead_threshold

    @property
    def name(self) -> str:
        return "orchestration"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        return [
            ScoreDimension(
                name="delegation_efficiency",
                value=self._compute_delegation_efficiency(episode),
                max_value=1.0,
                source=self.name,
            ),
            ScoreDimension(
                name="tool_strategy",
                value=self._compute_tool_strategy(episode),
                max_value=1.0,
                source=self.name,
            ),
            ScoreDimension(
                name="coordination_overhead",
                value=self._compute_coordination_overhead(episode),
                max_value=1.0,
                source=self.name,
            ),
            ScoreDimension(
                name="recovery_effectiveness",
                value=self._compute_recovery_effectiveness(episode),
                max_value=1.0,
                source=self.name,
            ),
            ScoreDimension(
                name="termination_quality",
                value=self._compute_termination_quality(episode),
                max_value=1.0,
                source=self.name,
            ),
        ]

    def detect_issues(self, episode: Episode) -> list[Issue]:
        issues: list[Issue] = []

        # Idle agents
        idle_agents = self._find_idle_agents(episode)
        for agent_name in idle_agents:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="Orchestration",
                    description=f"Idle agent: {agent_name}",
                )
            )

        # Tool retry loops
        retry_loops = self._find_retry_loops(episode)
        for tool_name, count in retry_loops:
            issues.append(
                Issue(
                    severity=Severity.ERROR,
                    category="Orchestration",
                    description=f"Tool retry loop: {tool_name} ({count} retries)",
                )
            )

        # High coordination overhead
        productive_ratio = self._compute_coordination_overhead(episode)
        if productive_ratio < self.overhead_threshold and len(episode.steps) > 0:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="Orchestration",
                    description=(
                        f"High coordination overhead: only {productive_ratio:.0%} "
                        "of steps are productive"
                    ),
                )
            )

        # Unrecovered failures
        if self._has_unrecovered_failures(episode):
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="Orchestration",
                    description="Unrecovered tool failure",
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Dimension computations
    # ------------------------------------------------------------------

    def _compute_delegation_efficiency(self, episode: Episode) -> float:
        """Ratio of agents with productive steps to total agents.

        Framework agents (tagged with ``metadata["framework_agent"]``)
        are excluded from both the numerator and denominator.
        """
        all_agents: set[str] = set()
        productive_agents: set[str] = set()

        for step in episode.steps:
            if step.agent_name and not step.metadata.get("framework_agent"):
                all_agents.add(step.agent_name)
                if step.kind in _PRODUCTIVE_KINDS or (
                    step.kind == StepKind.MESSAGE
                    and step.content
                    and len(step.content) > self.min_productive_message_len
                ):
                    productive_agents.add(step.agent_name)

        if not all_agents:
            return 1.0
        return len(productive_agents) / len(all_agents)

    def _compute_tool_strategy(self, episode: Episode) -> float:
        """Measure tool selection quality."""
        tool_calls = episode.steps_by_kind(StepKind.TOOL_CALL)
        if not tool_calls:
            return 1.0

        total_penalty = 0.0
        max_penalty = 3.0  # 3 penalty categories

        # (a) Repeated failed tool calls (same tool retried without change)
        failed_retries = self._count_failed_retries(tool_calls)
        if failed_retries > 0:
            total_penalty += min(1.0, failed_retries / 3.0)

        # (b) Tool diversity relative to task complexity
        unique_tools = {s.tool_name for s in tool_calls if s.tool_name}
        if len(tool_calls) > 3 and len(unique_tools) < 2:
            total_penalty += 1.0
        elif len(tool_calls) > 6 and len(unique_tools) < 3:
            total_penalty += 0.5

        # (c) Redundant tool calls (same tool + same args twice)
        redundant = self._count_redundant_calls(tool_calls)
        if redundant > 0:
            total_penalty += min(1.0, redundant / 3.0)

        return max(0.0, 1.0 - total_penalty / max_penalty)

    def _compute_coordination_overhead(self, episode: Episode) -> float:
        """Ratio of productive steps to total steps.

        A step is productive if it is a tool call, tool result, or fact
        check, **or** if it is a MESSAGE with substantive content (length
        above ``min_productive_message_len``).  Framework-agent messages
        are never counted as productive.
        """
        if not episode.steps:
            return 1.0
        productive = 0
        for s in episode.steps:
            if s.kind in _PRODUCTIVE_KINDS:
                productive += 1
            elif (
                s.kind == StepKind.MESSAGE
                and s.content
                and len(s.content) > self.min_productive_message_len
                and not s.metadata.get("framework_agent")
            ):
                productive += 1
        return productive / len(episode.steps)

    def _compute_recovery_effectiveness(self, episode: Episode) -> float:
        """Measure recovery from tool failures."""
        failures = []
        for i, step in enumerate(episode.steps):
            if step.kind == StepKind.TOOL_CALL and step.tool_succeeded is False:
                failures.append(i)

        if not failures:
            return 1.0

        recoveries = 0
        for fail_idx in failures:
            # Look for a successful step after the failure
            for j in range(fail_idx + 1, len(episode.steps)):
                subsequent = episode.steps[j]
                if subsequent.kind == StepKind.TOOL_CALL and subsequent.tool_succeeded:
                    recoveries += 1
                    break

        return recoveries / len(failures)

    def _compute_termination_quality(self, episode: Episode) -> float:
        """Detect wasted steps after the substantive answer is reached."""
        if not episode.steps:
            return 1.0

        # Find the last step that contributed new information
        last_productive_idx = -1
        for i in range(len(episode.steps) - 1, -1, -1):
            step = episode.steps[i]
            if step.kind in _PRODUCTIVE_KINDS:
                last_productive_idx = i
                break
            if step.kind == StepKind.MESSAGE and step.content:
                # Check if this looks like the final answer
                content_lower = (step.content or "").lower()
                if any(
                    marker in content_lower
                    for marker in ["answer", "result", "conclusion", "summary", "final"]
                ):
                    last_productive_idx = i
                    break

        if last_productive_idx == -1:
            return 1.0

        total = len(episode.steps)
        wasted = total - last_productive_idx - 1
        return max(0.0, 1.0 - wasted / total)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_idle_agents(self, episode: Episode) -> list[str]:
        """Find agents with no productive steps.

        Framework agents are excluded — they are expected to have
        minimal visible activity.
        """
        all_agents: set[str] = set()
        productive_agents: set[str] = set()

        for step in episode.steps:
            if step.agent_name and not step.metadata.get("framework_agent"):
                all_agents.add(step.agent_name)
                if step.kind in _PRODUCTIVE_KINDS or (
                    step.kind == StepKind.MESSAGE
                    and step.content
                    and len(step.content) > self.min_productive_message_len
                ):
                    productive_agents.add(step.agent_name)

        return sorted(all_agents - productive_agents)

    def _find_retry_loops(
        self, episode: Episode
    ) -> list[tuple[str, int]]:
        """Find tools retried 3+ times consecutively after failure."""
        results: list[tuple[str, int]] = []
        tool_calls = episode.steps_by_kind(StepKind.TOOL_CALL)

        consecutive_fails: dict[str, int] = defaultdict(int)
        reported: set[str] = set()

        for step in tool_calls:
            tool_name = step.tool_name or "unknown"
            if step.tool_succeeded is False:
                consecutive_fails[tool_name] += 1
                if consecutive_fails[tool_name] >= 3 and tool_name not in reported:
                    results.append((tool_name, consecutive_fails[tool_name]))
                    reported.add(tool_name)
            else:
                consecutive_fails[tool_name] = 0

        return results

    def _count_failed_retries(self, tool_calls: list[Step]) -> int:
        """Count failed tool calls where the same tool failed previously."""
        count = 0
        prev_failed: set[str] = set()
        for step in tool_calls:
            tool_name = step.tool_name or ""
            if step.tool_succeeded is False:
                if tool_name in prev_failed:
                    count += 1
                prev_failed.add(tool_name)
            else:
                prev_failed.discard(tool_name)
        return count

    def _count_redundant_calls(self, tool_calls: list[Step]) -> int:
        """Count tool calls with same name and args as a previous call."""
        seen: set[str] = set()
        redundant = 0
        for step in tool_calls:
            sig = f"{step.tool_name}:{step.tool_args}"
            if sig in seen:
                redundant += 1
            seen.add(sig)
        return redundant

    def _has_unrecovered_failures(self, episode: Episode) -> bool:
        """Check if any tool failure has no subsequent successful tool call."""
        failures = []
        for i, step in enumerate(episode.steps):
            if step.kind == StepKind.TOOL_CALL and step.tool_succeeded is False:
                failures.append(i)

        if not failures:
            return False

        for fail_idx in failures:
            recovered = False
            for j in range(fail_idx + 1, len(episode.steps)):
                if (
                    episode.steps[j].kind == StepKind.TOOL_CALL
                    and episode.steps[j].tool_succeeded
                ):
                    recovered = True
                    break
            if not recovered:
                return True
        return False
