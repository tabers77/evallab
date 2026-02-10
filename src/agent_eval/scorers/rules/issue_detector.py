"""Rule-based issue detection scorer.

Ported from log_evaluator.py:_detect_issues — detects 6 categories of
issues from an Episode's steps and metadata.
"""

from __future__ import annotations

from agent_eval.core.models import Episode, StepKind
from agent_eval.core.score import Issue, ScoreDimension, Severity


class IssueDetectorScorer:
    """Detects issues across 6 categories.

    Categories:
      1. Error Detection (exceptions, tool failures)
      2. Answer Quality (missing/short final answer)
      3. Agent Coordination (stalls, imbalance, insufficient delegation)
      4. Tool Usage (no tools, limited diversity, high failure rate)
      5. Efficiency (long execution, excessive LLM calls)
      6. Data Accuracy (no data, fact-check failures, numeric fabrications)

    Parameters
    ----------
    max_execution_seconds
        Threshold for a "long execution" warning.
    max_llm_calls
        Threshold for an "excessive LLM calls" warning.
    tool_failure_rate_threshold
        Tool failure rate above this triggers an error.
    """

    def __init__(
        self,
        max_execution_seconds: float = 300,
        max_llm_calls: int = 30,
        tool_failure_rate_threshold: float = 0.3,
    ) -> None:
        self.max_execution_seconds = max_execution_seconds
        self.max_llm_calls = max_llm_calls
        self.tool_failure_rate_threshold = tool_failure_rate_threshold

    @property
    def name(self) -> str:
        return "issue_detector"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        """Score is the inverse of the issue severity sum, normalized.

        A perfect episode with no issues scores 1.0.
        """
        issues = self.detect_issues(episode)
        penalty = sum(_severity_weight(i.severity) for i in issues)
        value = max(0.0, 1.0 - penalty / 100.0)
        return [
            ScoreDimension(
                name="issue_free",
                value=value,
                max_value=1.0,
                source=self.name,
            )
        ]

    def detect_issues(self, episode: Episode) -> list[Issue]:
        issues: list[Issue] = []
        issues.extend(self._detect_error_issues(episode))
        issues.extend(self._detect_answer_quality(episode))
        issues.extend(self._detect_coordination_issues(episode))
        issues.extend(self._detect_tool_usage_issues(episode))
        issues.extend(self._detect_efficiency_issues(episode))
        issues.extend(self._detect_data_accuracy_issues(episode))
        return issues

    # ------------------------------------------------------------------
    # Category 1: Error Detection
    # ------------------------------------------------------------------

    def _detect_error_issues(self, episode: Episode) -> list[Issue]:
        issues: list[Issue] = []
        raw_content = episode.metadata.get("raw_content", "")

        for i, line in enumerate(raw_content.split("\n")):
            if "Exception" in line or "Traceback" in line:
                issues.append(
                    Issue(
                        severity=Severity.CRITICAL,
                        category="Error Detection",
                        description="Exception/Traceback found",
                        line_number=i + 1,
                        context=line[:200],
                    )
                )
            if "Tool call failed" in line or "ToolExecutionError" in line:
                issues.append(
                    Issue(
                        severity=Severity.ERROR,
                        category="Tool Usage",
                        description="Tool execution failed",
                        line_number=i + 1,
                        context=line[:200],
                    )
                )
            if "API error" in line or "Connection error" in line or "Timeout" in line:
                issues.append(
                    Issue(
                        severity=Severity.ERROR,
                        category="Data Accuracy",
                        description="External API/data source error",
                        line_number=i + 1,
                        context=line[:200],
                    )
                )
            if "WARNING" in line:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        category="Error Detection",
                        description="Warning logged",
                        line_number=i + 1,
                        context=line[:200],
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # Category 2: Answer Quality
    # ------------------------------------------------------------------

    def _detect_answer_quality(self, episode: Episode) -> list[Issue]:
        issues: list[Issue] = []
        if episode.final_answer is None:
            issues.append(
                Issue(
                    severity=Severity.CRITICAL,
                    category="Answer Quality",
                    description="No final answer detected in logs",
                )
            )
        elif len(episode.final_answer) < 100:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="Answer Quality",
                    description=f"Final answer is very short ({len(episode.final_answer)} chars)",
                )
            )
        return issues

    # ------------------------------------------------------------------
    # Category 3: Agent Coordination
    # ------------------------------------------------------------------

    def _detect_coordination_issues(self, episode: Episode) -> list[Issue]:
        issues: list[Issue] = []

        # Agent turn counts (excluding orchestrator-like steps)
        agent_turns: dict[str, int] = {}
        for step in episode.steps:
            if step.kind == StepKind.MESSAGE and step.agent_name:
                agent_turns[step.agent_name] = agent_turns.get(step.agent_name, 0) + 1

        # Check for stalls (repeated tool calls)
        tool_steps = episode.steps_by_kind(StepKind.TOOL_CALL)
        stalls = 0
        previous: list[str] = []
        for ts in tool_steps:
            sig = f"{ts.agent_name}:{ts.tool_name}"
            if sig in previous[-3:]:
                stalls += 1
            previous.append(sig)

        if stalls > 3:
            issues.append(
                Issue(
                    severity=Severity.ERROR,
                    category="Agent Coordination",
                    description=f"Multiple stalls detected ({stalls} repeated action patterns)",
                )
            )

        # Agent imbalance
        non_orchestrator = {k: v for k, v in agent_turns.items() if v > 0}
        if len(non_orchestrator) > 1:
            max_turns = max(non_orchestrator.values())
            min_turns = min(non_orchestrator.values())
            if max_turns > min_turns * 5:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        category="Agent Coordination",
                        description=f"Agent turn imbalance detected (max: {max_turns}, min: {min_turns})",
                    )
                )

        # Insufficient delegation
        active_agents = episode.agents
        if len(active_agents) < 2:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="Agent Coordination",
                    description="Only one agent active - no delegation occurred",
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Category 4: Tool Usage
    # ------------------------------------------------------------------

    def _detect_tool_usage_issues(self, episode: Episode) -> list[Issue]:
        issues: list[Issue] = []
        tool_steps = episode.steps_by_kind(StepKind.TOOL_CALL)
        total = len(tool_steps)
        unique_tools = {s.tool_name for s in tool_steps if s.tool_name}
        failed = sum(1 for s in tool_steps if s.tool_succeeded is False)

        if total == 0:
            issues.append(
                Issue(
                    severity=Severity.CRITICAL,
                    category="Tool Usage",
                    description="No tools were called - likely hallucination",
                )
            )
        elif len(unique_tools) < 2 and total > 2:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="Tool Usage",
                    description=f"Limited tool diversity ({len(unique_tools)} unique tools)",
                )
            )

        if total > 0:
            failure_rate = failed / total
            if failure_rate > self.tool_failure_rate_threshold:
                issues.append(
                    Issue(
                        severity=Severity.ERROR,
                        category="Tool Usage",
                        description=f"High tool failure rate ({failure_rate:.1%})",
                    )
                )

        return issues

    # ------------------------------------------------------------------
    # Category 5: Efficiency
    # ------------------------------------------------------------------

    def _detect_efficiency_issues(self, episode: Episode) -> list[Issue]:
        issues: list[Issue] = []

        duration = episode.duration_seconds
        if duration and duration > self.max_execution_seconds:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="Efficiency",
                    description=f"Long execution time ({duration:.0f}s)",
                )
            )

        llm_steps = episode.steps_by_kind(StepKind.LLM_CALL)
        if len(llm_steps) > self.max_llm_calls:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="Efficiency",
                    description=f"Excessive LLM calls ({len(llm_steps)})",
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Category 6: Data Accuracy
    # ------------------------------------------------------------------

    def _detect_data_accuracy_issues(self, episode: Episode) -> list[Issue]:
        issues: list[Issue] = []
        raw_content = episode.metadata.get("raw_content", "")

        if "Your parameter selection returned no data" in raw_content:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="Data Accuracy",
                    description="Tool queries returned no data",
                )
            )

        # Fact-check failures
        fact_checks = episode.steps_by_kind(StepKind.FACT_CHECK)
        failures = sum(1 for fc in fact_checks if fc.metadata.get("verdict") == "FAIL")
        if failures > 0:
            total_fc = len(fact_checks)
            issues.append(
                Issue(
                    severity=Severity.ERROR,
                    category="Data Accuracy",
                    description=f"Fact checking detected {failures} issue(s) or hallucination(s)",
                    context=f"{failures} FAIL verdict(s) out of {total_fc} total fact-checks",
                )
            )

        return issues


def _severity_weight(severity: Severity) -> float:
    return {
        Severity.CRITICAL: 25.0,
        Severity.ERROR: 10.0,
        Severity.WARNING: 5.0,
        Severity.INFO: 0.0,
    }.get(severity, 0.0)
