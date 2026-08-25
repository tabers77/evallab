"""Tests for LLMJudgeScorer._build_transcript — what the judge is actually shown.

Regression tests for a defect that blinded the judge for a whole collection
period in a real deployment. Two independent causes:

1. ``TOOL_RESULT`` steps were not rendered at all, so any adapter that emits a
   TOOL_CALL/TOOL_RESULT *pair* (rather than putting everything on the call)
   handed the judge a transcript of empty tool lines.
2. ``tool_succeeded is None`` — the ``Step`` default, meaning "not recorded" —
   rendered as ``FAILED``.

Together the judge saw ``[Tool: x] (FAILED) ->`` for every tool call in every
episode and scored ``groundedness`` 0.00-0.25 on answers whose figures were
fully sourced, stating in its justifications that "both relevant tools failed
and no successful tool output is shown".
"""

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.scorers.llm_judge.judge import (
    LLMJudgeScorer,
    _result_text,
    _tool_status,
)


def _episode(steps: list[Step]) -> Episode:
    return Episode(
        episode_id="test",
        steps=steps,
        source_framework="test",
        final_answer="An answer.",
        metadata={},
    )


def _transcript(steps: list[Step]) -> str:
    return LLMJudgeScorer(llm_fn=lambda **_: "")._build_transcript(_episode(steps))


def _call(tool="get_finances", result=None, succeeded=None) -> Step:
    return Step(
        kind=StepKind.TOOL_CALL,
        agent_id="orchestrator",
        agent_name="orchestrator",
        tool_name=tool,
        tool_result=result,
        tool_succeeded=succeeded,
    )


def _result(tool="get_finances", content=None, succeeded=None) -> Step:
    return Step(
        kind=StepKind.TOOL_RESULT,
        agent_id="orchestrator",
        agent_name="orchestrator",
        tool_name=tool,
        content=content,
        tool_succeeded=succeeded,
    )


class TestToolStatus:
    def test_unknown_is_not_failed(self):
        """The `Step` default must not be reported as a failure."""
        assert _tool_status(None) == "?"

    def test_true_is_ok(self):
        assert _tool_status(True) == "OK"

    def test_false_is_failed(self):
        assert _tool_status(False) == "FAILED"


class TestResultText:
    def test_reads_tool_result(self):
        assert _result_text(_call(result="rows: 72")) == "rows: 72"

    def test_falls_back_to_content(self):
        assert _result_text(_result(content="rows: 72")) == "rows: 72"

    def test_truncates(self):
        assert len(_result_text(_call(result="x" * 5000))) == 200

    def test_empty_when_neither_present(self):
        assert _result_text(_call()) == ""


class TestSplitCallResultPair:
    """The adapter style that used to lose every tool result."""

    def test_result_from_a_separate_step_reaches_the_transcript(self):
        text = _transcript([
            _call(succeeded=True),
            _result(content="CONTRIBUTIONMARGIN=94384457.24", succeeded=True),
        ])
        assert "CONTRIBUTIONMARGIN=94384457.24" in text

    def test_pair_collapses_to_one_line(self):
        text = _transcript([
            _call(succeeded=True),
            _result(content="rows: 72", succeeded=True),
        ])
        assert text.count("get_finances") == 1
        assert text == "[Tool: get_finances] (OK) -> rows: 72"

    def test_successful_pair_is_not_reported_as_failed(self):
        text = _transcript(
            [_call(succeeded=True), _result(content="ok", succeeded=True)]
        )
        assert "FAILED" not in text

    def test_failure_on_the_result_step_is_reported(self):
        text = _transcript([
            _call(succeeded=True),
            _result(content="Error: relation does not exist", succeeded=False),
        ])
        assert "(FAILED)" in text
        assert "relation does not exist" in text

    def test_unrecorded_success_renders_unknown_not_failed(self):
        text = _transcript([_call(), _result(content="rows: 72")])
        assert "(?)" in text
        assert "FAILED" not in text

    def test_orphan_result_still_renders(self):
        """A TOOL_RESULT with no preceding call must not vanish."""
        text = _transcript([_result(content="rows: 72", succeeded=True)])
        assert "rows: 72" in text

    def test_mismatched_tool_name_does_not_overwrite(self):
        text = _transcript([
            _call(tool="tool_a", succeeded=True),
            _result(tool="tool_b", content="b output", succeeded=True),
        ])
        assert "tool_a" in text
        assert "b output" in text


class TestSelfContainedCallStep:
    """The other adapter style must keep working unchanged."""

    def test_result_on_the_call_step(self):
        text = _transcript([_call(result="rows: 72", succeeded=True)])
        assert text == "[Tool: get_finances] (OK) -> rows: 72"

    def test_a_following_result_step_does_not_duplicate_the_line(self):
        text = _transcript([
            _call(result="rows: 72", succeeded=True),
            _result(content="rows: 72", succeeded=True),
        ])
        assert text.count("rows: 72") == 2  # two distinct steps, both shown
        assert text.count("[Tool: get_finances]") == 1


class TestOtherStepKinds:
    def test_messages_and_fact_checks_survive(self):
        text = _transcript([
            Step(
                kind=StepKind.MESSAGE,
                agent_id="user",
                agent_name="user",
                content="Q?",
            ),
            _call(succeeded=True),
            _result(content="rows: 72", succeeded=True),
            Step(
                kind=StepKind.FACT_CHECK,
                agent_id="orchestrator",
                agent_name="orchestrator",
                content="PASS",
                metadata={"verdict": "PASS"},
            ),
        ])
        assert "[user]: Q?" in text
        assert "rows: 72" in text
        assert "[FactCheck: orchestrator] PASS" in text

    def test_a_message_between_call_and_result_breaks_the_pairing(self):
        """Pairing must not reach across an intervening turn."""
        text = _transcript([
            _call(succeeded=True),
            Step(kind=StepKind.MESSAGE, agent_id="a", agent_name="a", content="hm"),
            _result(content="late", succeeded=True),
        ])
        assert "[Tool result: get_finances] (OK) -> late" in text
