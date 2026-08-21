"""Reward-integrity probe — padding and tool-spam must not raise the score.

Written BEFORE the fix, deliberately. The intelligence-platform audit
(2026-08-17, finding F4) claimed ``RuleBasedScorer`` awards bonus points for
answer length and tool-call diversity, which makes verbosity and busywork
profitable — reward hacking built into the instrument. The point of landing
this probe first is to *prove* that claim rather than assume it: on the
pre-fix scorer these tests fail, which is the evidence the bonuses were the
defect.

It was also observed live before the fix. Two real runs through the
intelligence-platform conversation path scored:

    answer   1,047 chars, few tools  -> 97
    answer   8,448 chars, many tools -> 100

The longer, more tool-heavy answer scored higher on the same instrument.

The probe stays in the suite afterwards as a regression guard: the next
well-meaning "reward a thorough answer" bonus has to get past it.

Every comparison holds the issue set FIXED and non-empty. That is deliberate:
the scorer clamps to [0, 100] and the base is 100, so with no issues all
bonuses vanish into the ceiling and the probe would pass on the broken
scorer. The bug is only observable below the ceiling.

Deliberately NOT covered: ``zero_failure_bonus``. Rewarding "no tool call
failed" is not reward hacking — it cannot be farmed by padding, and the audit
named only the length and diversity bonuses.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.scorers.rules.deduction import RuleBasedScorer

_TS = datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc)


def _tool_step(name: str, ok: bool = True) -> Step:
    return Step(
        kind=StepKind.TOOL_CALL,
        agent_id="a",
        agent_name="Agent",
        tool_name=name,
        tool_args={},
        tool_result={"value": 1},
        tool_succeeded=ok,
        timestamp=_TS,
    )


def _episode(answer: str, tools: list[str]) -> Episode:
    return Episode(
        episode_id="probe",
        source_framework="probe",
        steps=[
            Step(
                kind=StepKind.MESSAGE,
                agent_id="a",
                agent_name="Agent",
                content="working",
                timestamp=_TS,
            ),
            *[_tool_step(t) for t in tools],
        ],
        final_answer=answer,
    )


def _issues(n: int = 2) -> list:
    """A fixed set of WARNINGs, identical across every comparison.

    Load-bearing: with zero issues the base score is already 100 and the
    scorer clamps to [0, 100], so every bonus is invisible. A probe run with
    no issues therefore PASSES on the buggy scorer and proves nothing — that
    was this file's first draft. Bonuses only become observable once a
    deduction has pulled the score below the ceiling, which is exactly the
    real-world case: the live 97-vs-100 pair had deductions in play.
    """
    from agent_eval.core.score import Issue, Severity

    return [
        Issue(severity=Severity.WARNING, category="probe", description=f"w{i}")
        for i in range(n)
    ]


def _score(episode: Episode) -> float:
    """Overall score with a FIXED deduction, so bonuses are observable."""
    dims = RuleBasedScorer().score_with_issues(episode, _issues())
    overall = next(d for d in dims if d.name == "overall_score")
    return overall.value


# The same substantive answer, padded. Identical information content.
_LEAN_ANSWER = "Revenue grew 12% to 4.2M, driven by the Nordic segment."
_PADDED_ANSWER = _LEAN_ANSWER + " " + ("Furthermore, it is worth noting. " * 40)


class TestVerbosityIsNotRewarded:
    """Padding an answer must not increase its score."""

    def test_padded_answer_does_not_outscore_lean_answer(self):
        lean = _score(_episode(_LEAN_ANSWER, ["get_revenue"]))
        padded = _score(_episode(_PADDED_ANSWER, ["get_revenue"]))

        assert padded <= lean, (
            f"Padding the same answer from {len(_LEAN_ANSWER)} to "
            f"{len(_PADDED_ANSWER)} characters raised the score from {lean} "
            f"to {padded}. The instrument pays for verbosity, so any agent "
            f"optimising against it learns to pad (audit F4)."
        )

    def test_length_alone_cannot_move_the_score(self):
        """Two answers differing only in padding must score identically."""
        short = _score(_episode("Revenue grew 12%.", ["get_revenue"]))
        long_ = _score(_episode("Revenue grew 12%." + " x" * 600, ["get_revenue"]))
        assert short == long_, (
            f"Answer length changed the score ({short} vs {long_}) with no "
            f"change in substance."
        )


class TestToolSpamIsNotRewarded:
    """Calling more distinct tools must not increase the score."""

    def test_tool_spam_does_not_outscore_lean_trajectory(self):
        lean = _score(_episode(_LEAN_ANSWER, ["get_revenue"]))
        spam = _score(
            _episode(
                _LEAN_ANSWER,
                ["get_revenue", "list_dims", "get_meta", "list_cats", "get_cfg"],
            )
        )
        assert spam <= lean, (
            f"Reaching for 5 distinct tools instead of 1 raised the score "
            f"from {lean} to {spam} for the same answer. The instrument pays "
            f"for busywork (audit F4)."
        )


class TestBothLeversTogether:
    """The live-observed shape: long answer + many tools."""

    def test_padded_and_tool_heavy_does_not_outscore_lean(self):
        lean = _score(_episode(_LEAN_ANSWER, ["get_revenue"]))
        hacked = _score(
            _episode(
                _PADDED_ANSWER,
                ["get_revenue", "list_dims", "get_meta", "list_cats", "get_cfg"],
            )
        )
        assert hacked <= lean, (
            f"A padded, tool-heavy episode scored {hacked} against {lean} for "
            f"the lean equivalent — reproducing the 97 vs 100 gap observed on "
            f"two live production runs before the fix."
        )


class TestLegitimateSignalsStillWork:
    """Removing the bonuses must not flatten the scorer entirely."""

    @pytest.mark.parametrize(
        ("severity_name", "expected_less_than"),
        [("CRITICAL", 100.0), ("ERROR", 100.0), ("WARNING", 100.0)],
    )
    def test_issues_still_deduct(self, severity_name, expected_less_than):
        from agent_eval.core.score import Issue, Severity

        issue = Issue(
            severity=getattr(Severity, severity_name),
            category="probe",
            description="x",
        )
        dims = RuleBasedScorer().score_with_issues(
            _episode(_LEAN_ANSWER, ["get_revenue"]), [issue]
        )
        overall = next(d for d in dims if d.name == "overall_score")
        assert overall.value < expected_less_than, (
            "Deductions stopped working — the probe must not be satisfied by "
            "making every score constant."
        )
