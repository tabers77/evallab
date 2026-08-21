"""Judge integrity — abstention, single-call, keyword contract, justifications.

Covers audit findings F5, F6, F7 and F8 from the intelligence-platform
evallab audit (2026-08-17), all landed in 0.3.0. The F4 reward-hacking probe
lives separately in ``tests/scorers/rules/test_reward_hacking_probe.py``.

Each finding here was a way the judge could report a number that did not mean
what a reader would assume it meant.
"""

from __future__ import annotations

import json

import pytest

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import ScoreVector
from agent_eval.scorers.llm_judge.judge import LLMJudgeScorer
from agent_eval.scorers.llm_judge.prompts import DEFAULT_DIMENSIONS


def _episode(eid: str = "judge-1") -> Episode:
    return Episode(
        episode_id=eid,
        source_framework="test",
        steps=[
            Step(
                kind=StepKind.MESSAGE,
                agent_id="a",
                agent_name="Agent",
                content="analysis",
            )
        ],
        final_answer="Revenue grew 12%.",
    )


def _good_response(score: float = 0.8, justification: str = "Solid.") -> str:
    return json.dumps(
        {
            "evaluations": [
                {"dimension": name, "score": score, "justification": justification}
                for name in DEFAULT_DIMENSIONS
            ]
        }
    )


# ---------------------------------------------------------------------------
# F7 — keyword-only callable
# ---------------------------------------------------------------------------


class TestKeywordOnlyContract:
    """The platform declared (user_prompt, system_prompt) and got them swapped.

    Positional order silently tolerated that. Naming the arguments makes the
    mismatch impossible to express.
    """

    def test_llm_fn_receives_named_arguments(self):
        seen: dict[str, str] = {}

        def llm_fn(*, system_prompt: str, user_prompt: str) -> str:
            seen["system"] = system_prompt
            seen["user"] = user_prompt
            return _good_response()

        LLMJudgeScorer(llm_fn=llm_fn).score(_episode())

        assert seen, "llm_fn was never called"
        # The judge's own instruction goes in the system slot; the transcript
        # and the dimension questions go in the user slot. Swapping these is
        # exactly the bug (F7).
        assert "judge" in seen["system"].lower() or "evaluat" in seen["system"].lower()
        assert "Revenue grew 12%." in seen["user"]

    def test_positional_only_callable_is_not_silently_accepted(self):
        """A legacy positional fake must abstain, not appear to work."""

        def legacy(system_prompt, user_prompt, /):  # positional-only
            return _good_response()

        dims = LLMJudgeScorer(llm_fn=legacy).score(_episode())
        assert all(d.abstained for d in dims), (
            "A callable that cannot accept keyword arguments produced scores. "
            "The keyword contract is not being enforced."
        )


# ---------------------------------------------------------------------------
# F6 — one judge call per episode
# ---------------------------------------------------------------------------


class TestJudgeRunsOncePerEpisode:
    """score() and detect_issues() each used to invoke the LLM."""

    def test_score_then_detect_issues_makes_one_call(self):
        calls: list[int] = []

        def llm_fn(*, system_prompt: str, user_prompt: str) -> str:
            calls.append(1)
            return _good_response()

        scorer = LLMJudgeScorer(llm_fn=llm_fn)
        ep = _episode()
        scorer.score(ep)
        scorer.detect_issues(ep)

        assert len(calls) == 1, (
            f"The judge ran {len(calls)} times for one episode. Every "
            f"evaluation pays double, and the issues can be derived from a "
            f"different response than the score (audit F6)."
        )

    def test_a_different_episode_is_not_served_from_cache(self):
        calls: list[str] = []

        def llm_fn(*, system_prompt: str, user_prompt: str) -> str:
            calls.append("x")
            return _good_response()

        scorer = LLMJudgeScorer(llm_fn=llm_fn)
        scorer.score(_episode("ep-a"))
        scorer.score(_episode("ep-b"))
        assert len(calls) == 2, "Memoisation leaked across episodes"


# ---------------------------------------------------------------------------
# F5 — abstain instead of scoring 0.0
# ---------------------------------------------------------------------------


class TestAbstentionNotZero:
    """A failed judge call is an absence of measurement, not a bad score."""

    def test_api_failure_abstains(self):
        def boom(*, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("503 upstream")

        dims = LLMJudgeScorer(llm_fn=boom).score(_episode())
        assert dims, "no dimensions returned at all"
        assert all(d.abstained for d in dims)
        assert all("abstained" in (d.justification or "") for d in dims)

    def test_unparseable_response_abstains(self):
        def garbage(*, system_prompt: str, user_prompt: str) -> str:
            return "I'm afraid I can't do that."

        dims = LLMJudgeScorer(llm_fn=garbage).score(_episode())
        assert all(d.abstained for d in dims)

    def test_abstention_is_excluded_from_the_aggregate(self):
        """This is the -15 points the audit measured."""

        def boom(*, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("timeout")

        abstained = LLMJudgeScorer(llm_fn=boom).score(_episode())
        real = [d for d in LLMJudgeScorer(llm_fn=lambda **k: _good_response(0.8)).score(
            _episode("ep-real")
        )]

        mixed = ScoreVector(episode_id="mix", dimensions=real + abstained)
        clean = ScoreVector(episode_id="clean", dimensions=real)

        assert mixed.overall == pytest.approx(clean.overall), (
            f"Abstained dimensions dragged the aggregate from "
            f"{clean.overall} to {mixed.overall}. A transient API error is "
            f"being reported as a quality drop (audit F5)."
        )

    def test_all_abstained_is_distinguishable_from_a_real_zero(self):
        def boom(*, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("down")

        sv = ScoreVector(
            episode_id="x", dimensions=LLMJudgeScorer(llm_fn=boom).score(_episode())
        )
        assert sv.all_abstained is True
        assert sv.overall == 0.0  # only because there is no honest number

        real_zero = ScoreVector(
            episode_id="y",
            dimensions=LLMJudgeScorer(
                llm_fn=lambda **k: _good_response(0.0)
            ).score(_episode("z")),
        )
        assert real_zero.all_abstained is False, (
            "A genuine all-zero evaluation is being reported as an abstention."
        )

    def test_abstention_does_not_raise_a_low_score_issue(self):
        """Otherwise F5's false signal returns dressed as an issue."""

        def boom(*, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("down")

        issues = LLMJudgeScorer(llm_fn=boom).detect_issues(_episode())
        assert issues == [], f"abstention produced issues: {issues}"


# ---------------------------------------------------------------------------
# F8 — retain per-dimension justifications
# ---------------------------------------------------------------------------


class TestJustificationsRetained:
    """The prompt always asked for these; the parser always discarded them."""

    def test_batch_justification_is_kept(self):
        dims = LLMJudgeScorer(
            llm_fn=lambda **k: _good_response(0.7, "Numbers trace to tool output.")
        ).score(_episode())
        assert dims
        assert all(d.justification == "Numbers trace to tool output." for d in dims), (
            "Justifications were dropped, so a stored score cannot be "
            "explained or disputed after the fact (audit F8)."
        )

    def test_missing_dimension_abstains_rather_than_scoring_zero(self):
        """A dimension the model omitted was never judged."""
        partial = json.dumps(
            {
                "evaluations": [
                    {
                        "dimension": next(iter(DEFAULT_DIMENSIONS)),
                        "score": 0.9,
                        "justification": "Good.",
                    }
                ]
            }
        )
        dims = LLMJudgeScorer(llm_fn=lambda **k: partial).score(_episode())
        scored = [d for d in dims if not d.abstained]
        abstained = [d for d in dims if d.abstained]
        assert len(scored) == 1
        assert abstained, "omitted dimensions were scored 0.0 instead of abstained"

    def test_justification_survives_serialization(self):
        dims = LLMJudgeScorer(
            llm_fn=lambda **k: _good_response(0.6, "Partially grounded.")
        ).score(_episode())
        payload = ScoreVector(episode_id="s", dimensions=dims).to_dict()
        assert all(
            d["justification"] == "Partially grounded." for d in payload["dimensions"]
        )
        assert all("abstained" in d for d in payload["dimensions"])
