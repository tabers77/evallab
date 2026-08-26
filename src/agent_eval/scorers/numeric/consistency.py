"""Numeric consistency scorer for hallucination detection.

Ported from numeric_validator.py:validate_numeric_consistency — compares
numbers in an agent's final answer against numbers from tool results.
"""

from __future__ import annotations

from agent_eval.core.models import Episode, StepKind
from agent_eval.core.score import Issue, ScoreDimension, Severity
from agent_eval.scorers.numeric.extraction import (
    extract_answer_block,
    extract_numbers_from_text,
    extract_numbers_from_tool_results,
    extract_numbers_with_context,
)


class NumericConsistencyScorer:
    """Detect fabricated numbers by comparing answers against tool results.

    Parameters
    ----------
    tolerance
        Acceptable relative error margin (default 5%).
    min_value
        Minimum absolute value to consider (skip trivially small numbers).
    """

    # Episode metadata key an adapter sets when the tool evidence in this
    # episode is knowingly incomplete — most commonly a conversation turn whose
    # answer draws on documents fetched in an *earlier* turn. Nothing this
    # scorer can see distinguishes "cited from last turn" from "invented", so
    # measuring is not possible and the honest answer is to abstain rather than
    # report every carried-over figure as a fabrication.
    EVIDENCE_INCOMPLETE_KEY = "evidence_incomplete"

    def __init__(self, tolerance: float = 0.05, min_value: float = 1.0) -> None:
        self.tolerance = tolerance
        self.min_value = min_value

    @property
    def name(self) -> str:
        return "numeric_consistency"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        """Return a score dimension for numeric accuracy.

        Value is the fraction of answer numbers that matched tool results.
        """
        if self._evidence_incomplete(episode):
            return [
                ScoreDimension(
                    name="numeric_accuracy",
                    value=0.0,
                    max_value=1.0,
                    source=self.name,
                    abstained=True,
                )
            ]

        fabrications = self._find_fabrications(episode)
        answer_numbers = self._get_answer_numbers(episode)

        total = len(answer_numbers)
        if total == 0:
            return [
                ScoreDimension(
                    name="numeric_accuracy",
                    value=1.0,
                    max_value=1.0,
                    source=self.name,
                )
            ]

        matched = total - len(fabrications)
        return [
            ScoreDimension(
                name="numeric_accuracy",
                value=matched / total,
                max_value=1.0,
                source=self.name,
            )
        ]

    def _evidence_incomplete(self, episode: Episode) -> bool:
        return bool((episode.metadata or {}).get(self.EVIDENCE_INCOMPLETE_KEY))

    def detect_issues(self, episode: Episode) -> list[Issue]:
        """Detect fabricated numbers as CRITICAL issues.

        Returns nothing when the episode declares its evidence incomplete: a
        figure carried over from an earlier conversation turn is unverifiable
        here by construction, and reporting it as fabricated was observed
        turning correct multi-turn answers into grade F.
        """
        if self._evidence_incomplete(episode):
            return []
        fabrications = self._find_fabrications(episode)
        issues: list[Issue] = []

        for fab in fabrications:
            closest = fab.get("closest_match")
            error_pct = fab.get("error_percent")

            context_parts = [f"Fabricated: {fab['number']:,.2f}"]
            if closest is not None:
                context_parts.append(f"Closest tool number: {closest:,.2f}")
            if error_pct is not None:
                context_parts.append(f"Error: {error_pct:.1f}%")
            else:
                context_parts.append("No match in tools")

            issues.append(
                Issue(
                    severity=Severity.CRITICAL,
                    category="Data Fabrication",
                    description=fab["description"],
                    context=", ".join(context_parts),
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_answer_numbers(self, episode: Episode) -> list[float]:
        """Extract numbers from the episode's final answer."""
        if episode.final_answer:
            return extract_numbers_from_text(episode.final_answer)

        # Fallback: look in raw content metadata
        content = episode.metadata.get("raw_content", "")
        answer_text = extract_answer_block(content)
        return extract_numbers_from_text(answer_text)

    def _get_tool_numbers(self, episode: Episode) -> dict[str, list[float]]:
        """Extract numbers from tool call steps."""
        raw_events = [
            s.metadata.get("raw_event", {})
            for s in episode.steps
            if s.kind == StepKind.TOOL_CALL
        ]
        # Also include events from metadata if available
        if not raw_events:
            raw_events = episode.metadata.get("raw_events", [])
        return extract_numbers_from_tool_results(raw_events)

    def _find_fabrications(self, episode: Episode) -> list[dict]:
        """Core validation: compare answer numbers against tool numbers.

        Numbers identified as derived or approximate (via surrounding text
        context) are not flagged as fabrications — they represent
        calculations, proposals, or estimates that are not expected to
        appear verbatim in tool output.
        """
        answer_text = episode.final_answer or ""
        if not answer_text:
            content = episode.metadata.get("raw_content", "")
            answer_text = extract_answer_block(content)

        answer_with_ctx = extract_numbers_with_context(answer_text)
        tool_numbers = self._get_tool_numbers(episode)

        all_tool_numbers: list[float] = []
        for nums in tool_numbers.values():
            all_tool_numbers.extend(nums)

        if not all_tool_numbers:
            return []

        issues: list[dict] = []

        for entry in answer_with_ctx:
            answer_num = entry["value"]
            if abs(answer_num) < self.min_value:
                continue

            # Skip derived/approximate numbers — these are calculations,
            # proposals, or estimates, not direct tool citations.
            if entry["is_derived"] or entry["in_percent_range"]:
                continue

            found = False
            closest = None
            closest_error = float("inf")

            for tool_num in all_tool_numbers:
                error = abs(answer_num - tool_num) / max(abs(tool_num), 1e-9)
                if error < closest_error:
                    closest = tool_num
                    closest_error = error
                if error <= self.tolerance:
                    found = True
                    break

            if not found:
                issues.append(
                    {
                        "type": "numeric_fabrication",
                        "number": answer_num,
                        "found_in_tools": False,
                        "closest_match": closest,
                        "error_percent": closest_error * 100 if closest else None,
                        "description": (
                            f"Number {answer_num:,.2f} not found in tool results "
                            f"(closest: {closest:,.2f}, error: {closest_error * 100:.1f}%)"
                            if closest
                            else f"Number {answer_num:,.2f} not found in tool results"
                        ),
                    }
                )

        return issues
