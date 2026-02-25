"""Intrinsic reasoning quality scorer.

Evaluates the reasoning capabilities exhibited in agent MESSAGE steps,
independent of orchestration quality. Based on the distinction between
post-training reasoning (intrinsic) and in-context reasoning from
"Agentic Reasoning for Large Language Models" (Wei et al., 2026).

Produces 4 dimensions: reasoning_depth, reasoning_coherence,
self_correction, and plan_quality.
"""

from __future__ import annotations

import re
from collections import defaultdict

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Issue, ScoreDimension, Severity

# Reasoning indicator patterns
_REASONING_MARKERS = re.compile(
    r"\b("
    r"therefore|because|since|consequently|thus|hence|"
    r"first[\s,].*(?:then|next|second)|"
    r"step\s+\d|"
    r"if\s+.*then|"
    r"this\s+(?:means|implies|suggests|indicates)|"
    r"as\s+a\s+result|"
    r"it\s+follows\s+that|"
    r"given\s+that|"
    r"in\s+order\s+to|"
    r"the\s+reason\s+is"
    r")\b",
    re.IGNORECASE,
)

_CONTRADICTION_NEGATIONS = re.compile(
    r"\b(is\s+not|isn't|aren't|won't|cannot|can't|doesn't|don't|no\s+\w+|never)\b",
    re.IGNORECASE,
)

_SELF_CORRECTION_PATTERNS = re.compile(
    r"\b("
    r"actually|correction|I\s+was\s+wrong|let\s+me\s+reconsider|"
    r"I\s+(?:need\s+to\s+)?correct|on\s+second\s+thought|"
    r"wait[\s,]|I\s+made\s+(?:a|an)\s+(?:error|mistake)|"
    r"let\s+me\s+re(?:think|vise|evaluate)|"
    r"scratch\s+that|disregard\s+(?:that|my\s+previous)"
    r")\b",
    re.IGNORECASE,
)

_PLAN_STEP_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\d+[\.\)]\s+|[-*]\s+|(?:first|second|third|next|then|finally)[,:\s])",
    re.IGNORECASE,
)

_VERB_PATTERN = re.compile(
    r"\b(?:get|find|search|create|update|check|verify|calculate|analyze|"
    r"compare|extract|fetch|retrieve|run|execute|send|build|generate|"
    r"determine|identify|review|evaluate|process|call|use|look)\b",
    re.IGNORECASE,
)

_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?(?:\s*%|[KMBkmb])?\b")


def _get_message_steps(episode: Episode) -> list[Step]:
    """Return MESSAGE steps with non-empty content."""
    return [
        s
        for s in episode.steps
        if s.kind == StepKind.MESSAGE and s.content and s.content.strip()
    ]


class IntrinsicReasoningScorer:
    """Scores the intrinsic reasoning quality of agent messages.

    Analyzes MESSAGE-level content to evaluate reasoning depth,
    coherence, self-correction, and planning quality.
    """

    @property
    def name(self) -> str:
        return "intrinsic_reasoning"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        msg_steps = _get_message_steps(episode)
        return [
            ScoreDimension(
                name="reasoning_depth",
                value=self._compute_reasoning_depth(msg_steps),
                max_value=1.0,
                source=self.name,
            ),
            ScoreDimension(
                name="reasoning_coherence",
                value=self._compute_reasoning_coherence(msg_steps),
                max_value=1.0,
                source=self.name,
            ),
            ScoreDimension(
                name="self_correction",
                value=self._compute_self_correction(msg_steps),
                max_value=1.0,
                source=self.name,
            ),
            ScoreDimension(
                name="plan_quality",
                value=self._compute_plan_quality(msg_steps),
                max_value=1.0,
                source=self.name,
            ),
        ]

    def detect_issues(self, episode: Episode) -> list[Issue]:
        issues: list[Issue] = []
        msg_steps = _get_message_steps(episode)

        # No reasoning markers in any message
        if msg_steps and self._compute_reasoning_depth(msg_steps) == 0.0:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="Intrinsic Reasoning",
                    description="Shallow reasoning: no reasoning markers found in any message",
                )
            )

        # Contradictions
        contradictions = self._find_contradictions(msg_steps)
        for agent_name, details in contradictions:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category="Intrinsic Reasoning",
                    description=f"Contradiction detected in {agent_name}'s messages",
                    context=details,
                )
            )

        # No planning in multi-step tasks
        total_steps = len(episode.steps)
        if total_steps > 5 and self._compute_plan_quality(msg_steps) == 0.0:
            issues.append(
                Issue(
                    severity=Severity.INFO,
                    category="Intrinsic Reasoning",
                    description="No explicit planning detected in multi-step task",
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Dimension computations
    # ------------------------------------------------------------------

    def _compute_reasoning_depth(self, msg_steps: list[Step]) -> float:
        """Ratio of message steps containing reasoning markers."""
        if not msg_steps:
            return 0.0
        with_markers = sum(
            1 for s in msg_steps if _REASONING_MARKERS.search(s.content or "")
        )
        return with_markers / len(msg_steps)

    def _compute_reasoning_coherence(self, msg_steps: list[Step]) -> float:
        """1 - (contradictions / total_agent_message_pairs)."""
        if len(msg_steps) < 2:
            return 1.0

        contradictions_found = len(self._find_contradictions(msg_steps))
        total_pairs = self._count_agent_message_pairs(msg_steps)
        if total_pairs == 0:
            return 1.0
        return max(0.0, 1.0 - contradictions_found / total_pairs)

    def _compute_self_correction(self, msg_steps: list[Step]) -> float:
        """Ratio of self-corrections to substantive message steps, capped at 1.0."""
        if not msg_steps:
            return 0.0
        # Substantive = content longer than 20 chars
        substantive = [s for s in msg_steps if len(s.content or "") > 20]
        if not substantive:
            return 0.0
        corrections = sum(
            1 for s in substantive if _SELF_CORRECTION_PATTERNS.search(s.content or "")
        )
        return min(1.0, corrections / len(substantive))

    def _compute_plan_quality(self, msg_steps: list[Step]) -> float:
        """Score based on plan presence, multi-step, and actionability."""
        if not msg_steps:
            return 0.0

        best_score = 0.0
        for step in msg_steps:
            content = step.content or ""
            plan_matches = _PLAN_STEP_PATTERN.findall(content)
            if not plan_matches:
                continue

            # (a) plan present at all → 0.33
            step_score = 1.0 / 3.0

            # (b) plan has 2+ steps → +0.33
            if len(plan_matches) >= 2:
                step_score += 1.0 / 3.0

                # (c) plan steps are actionable (contain verbs) → +0.34
                actionable = sum(
                    1
                    for m in plan_matches
                    if _VERB_PATTERN.search(content[content.find(m) :])
                )
                if actionable >= 2:
                    step_score += 1.0 / 3.0

            best_score = max(best_score, step_score)

        return min(1.0, best_score)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_contradictions(
        self, msg_steps: list[Step]
    ) -> list[tuple[str, str]]:
        """Find contradictions between sequential messages from the same agent."""
        contradictions: list[tuple[str, str]] = []

        # Group sequential messages by agent
        agent_messages: dict[str, list[str]] = defaultdict(list)
        for step in msg_steps:
            agent_messages[step.agent_name].append(step.content or "")

        for agent_name, messages in agent_messages.items():
            for i in range(len(messages) - 1):
                curr = messages[i]
                nxt = messages[i + 1]

                # Check for numeric contradictions
                curr_nums = set(_NUMBER_PATTERN.findall(curr))
                nxt_nums = set(_NUMBER_PATTERN.findall(nxt))
                if curr_nums and nxt_nums:
                    # Look for same context but different numbers
                    for num in curr_nums:
                        if num not in nxt_nums and _has_negation_of(curr, nxt):
                            contradictions.append(
                                (agent_name, f"'{curr[:80]}' vs '{nxt[:80]}'")
                            )
                            break

                # Check for assertion followed by negation
                if _has_negation_of(curr, nxt):
                    contradictions.append(
                        (agent_name, f"'{curr[:80]}' vs '{nxt[:80]}'")
                    )

        return contradictions

    def _count_agent_message_pairs(self, msg_steps: list[Step]) -> int:
        """Count total sequential message pairs per agent."""
        agent_counts: dict[str, int] = defaultdict(int)
        for step in msg_steps:
            agent_counts[step.agent_name] += 1
        return sum(max(0, c - 1) for c in agent_counts.values())


def _has_negation_of(text_a: str, text_b: str) -> bool:
    """Check if text_b negates a claim in text_a."""
    # Extract key phrases (simple noun phrases around verbs)
    a_lower = text_a.lower()
    b_lower = text_b.lower()

    # Look for direct negation patterns
    a_has_negation = bool(_CONTRADICTION_NEGATIONS.search(a_lower))
    b_has_negation = bool(_CONTRADICTION_NEGATIONS.search(b_lower))

    # If one has negation and the other doesn't, and they share key words
    if a_has_negation != b_has_negation:
        a_words = set(re.findall(r"\b\w{4,}\b", a_lower)) - {
            "this",
            "that",
            "with",
            "from",
            "have",
            "been",
        }
        b_words = set(re.findall(r"\b\w{4,}\b", b_lower)) - {
            "this",
            "that",
            "with",
            "from",
            "have",
            "been",
        }
        overlap = a_words & b_words
        if len(overlap) >= 2:
            return True
    return False
