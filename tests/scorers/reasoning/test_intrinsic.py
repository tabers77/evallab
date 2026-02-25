"""Tests for scorers.reasoning.intrinsic — IntrinsicReasoningScorer."""

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.protocols import Scorer
from agent_eval.core.score import Severity
from agent_eval.scorers.reasoning.intrinsic import IntrinsicReasoningScorer


def _make_episode(**kwargs) -> Episode:
    defaults = {
        "episode_id": "test",
        "steps": [],
        "source_framework": "test",
        "final_answer": "A detailed answer with enough content." * 5,
        "metadata": {},
    }
    defaults.update(kwargs)
    return Episode(**defaults)


def _msg(agent_name: str, content: str) -> Step:
    return Step(
        kind=StepKind.MESSAGE,
        agent_id=agent_name.lower(),
        agent_name=agent_name,
        content=content,
    )


class TestIntrinsicReasoningScorerBasics:
    def test_name(self):
        scorer = IntrinsicReasoningScorer()
        assert scorer.name == "intrinsic_reasoning"

    def test_protocol_compliance(self):
        scorer = IntrinsicReasoningScorer()
        assert isinstance(scorer, Scorer)

    def test_score_returns_four_dimensions(self):
        ep = _make_episode(steps=[_msg("Agent1", "Hello")])
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        assert len(dims) == 4
        names = {d.name for d in dims}
        assert names == {
            "reasoning_depth",
            "reasoning_coherence",
            "self_correction",
            "plan_quality",
        }

    def test_empty_episode(self):
        ep = _make_episode(steps=[])
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        assert all(d.value == 0.0 or d.value == 1.0 for d in dims)


class TestReasoningDepth:
    def test_with_reasoning_markers(self):
        ep = _make_episode(
            steps=[
                _msg("Agent1", "First, I need to analyze the data. Therefore the result is clear."),
                _msg("Agent1", "Because the revenue increased, we should adjust the forecast."),
            ]
        )
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        depth = next(d for d in dims if d.name == "reasoning_depth")
        assert depth.value == 1.0

    def test_without_reasoning_markers(self):
        ep = _make_episode(
            steps=[
                _msg("Agent1", "Hello world"),
                _msg("Agent1", "The data is here"),
            ]
        )
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        depth = next(d for d in dims if d.name == "reasoning_depth")
        assert depth.value == 0.0

    def test_partial_reasoning(self):
        ep = _make_episode(
            steps=[
                _msg("Agent1", "Therefore the result follows logically"),
                _msg("Agent1", "The data is here"),
                _msg("Agent1", "Processing complete"),
            ]
        )
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        depth = next(d for d in dims if d.name == "reasoning_depth")
        assert 0.0 < depth.value < 1.0

    def test_shallow_reasoning_issue(self):
        ep = _make_episode(
            steps=[
                _msg("Agent1", "Hello"),
                _msg("Agent1", "Done"),
            ]
        )
        scorer = IntrinsicReasoningScorer()
        issues = scorer.detect_issues(ep)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any("shallow reasoning" in i.description.lower() for i in warnings)


class TestReasoningCoherence:
    def test_no_contradictions(self):
        ep = _make_episode(
            steps=[
                _msg("Agent1", "The revenue is growing steadily this quarter"),
                _msg("Agent1", "The revenue growth continues to be positive"),
            ]
        )
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        coherence = next(d for d in dims if d.name == "reasoning_coherence")
        assert coherence.value == 1.0

    def test_with_contradictions(self):
        ep = _make_episode(
            steps=[
                _msg("Agent1", "The revenue trend is clearly positive and growing"),
                _msg("Agent1", "The revenue trend is not positive and isn't growing"),
            ]
        )
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        coherence = next(d for d in dims if d.name == "reasoning_coherence")
        assert coherence.value < 1.0

    def test_single_message_is_coherent(self):
        ep = _make_episode(steps=[_msg("Agent1", "Just one message")])
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        coherence = next(d for d in dims if d.name == "reasoning_coherence")
        assert coherence.value == 1.0

    def test_contradiction_issue_detected(self):
        ep = _make_episode(
            steps=[
                _msg("Agent1", "The revenue trend is clearly positive and growing"),
                _msg("Agent1", "The revenue trend is not positive and isn't growing"),
            ]
        )
        scorer = IntrinsicReasoningScorer()
        issues = scorer.detect_issues(ep)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any("contradiction" in i.description.lower() for i in warnings)


class TestSelfCorrection:
    def test_with_corrections(self):
        ep = _make_episode(
            steps=[
                _msg("Agent1", "Actually, I need to reconsider my previous analysis of the data"),
                _msg("Agent1", "Wait, let me rethink the approach to solving this problem"),
            ]
        )
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        correction = next(d for d in dims if d.name == "self_correction")
        assert correction.value > 0.0

    def test_without_corrections(self):
        ep = _make_episode(
            steps=[
                _msg("Agent1", "The analysis shows clear results here"),
                _msg("Agent1", "Moving on to the next step in processing"),
            ]
        )
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        correction = next(d for d in dims if d.name == "self_correction")
        assert correction.value == 0.0

    def test_short_messages_ignored(self):
        ep = _make_episode(
            steps=[_msg("Agent1", "Actually")]
        )
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        correction = next(d for d in dims if d.name == "self_correction")
        assert correction.value == 0.0


class TestPlanQuality:
    def test_with_planning_structure(self):
        ep = _make_episode(
            steps=[
                _msg(
                    "Agent1",
                    "1. Find the revenue data for Q1\n"
                    "2. Calculate the growth rate\n"
                    "3. Generate the final report",
                ),
            ]
        )
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        plan = next(d for d in dims if d.name == "plan_quality")
        assert plan.value > 0.5

    def test_without_planning(self):
        ep = _make_episode(
            steps=[_msg("Agent1", "Just doing some work here")]
        )
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        plan = next(d for d in dims if d.name == "plan_quality")
        assert plan.value == 0.0

    def test_no_planning_issue_in_multistep(self):
        steps = [_msg("Agent1", f"Message {i}") for i in range(3)]
        steps.extend(
            [
                Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="",
                    agent_name="",
                    tool_name=f"tool{i}",
                    tool_succeeded=True,
                )
                for i in range(4)
            ]
        )
        ep = _make_episode(steps=steps)
        scorer = IntrinsicReasoningScorer()
        issues = scorer.detect_issues(ep)
        info = [i for i in issues if i.severity == Severity.INFO]
        assert any("no explicit planning" in i.description.lower() for i in info)

    def test_single_step_plan(self):
        ep = _make_episode(
            steps=[_msg("Agent1", "1. Check the database for results")]
        )
        scorer = IntrinsicReasoningScorer()
        dims = scorer.score(ep)
        plan = next(d for d in dims if d.name == "plan_quality")
        # Single step plan: present but not multi-step
        assert 0.0 < plan.value <= 0.5
