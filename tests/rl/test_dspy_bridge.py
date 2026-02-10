"""Tests for rl.dspy_bridge — DSPyMetricBridge."""

import pytest

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Issue, ScoreDimension, ScoreVector, Severity
from agent_eval.rl.dspy_bridge import DSPyMetricBridge, _default_episode_builder


class FakeScorer:
    @property
    def name(self):
        return "fake"

    def score(self, episode):
        return [ScoreDimension(name="quality", value=0.8, max_value=1.0)]

    def detect_issues(self, episode):
        return []


class FakeRewardFn:
    def compute(self, score_vector):
        return score_vector.overall


class TestDefaultEpisodeBuilder:
    def test_builds_episode(self):
        ep = _default_episode_builder("What is 2+2?", "4")
        assert ep.source_framework == "dspy"
        assert len(ep.steps) == 2
        assert ep.steps[0].agent_name == "user"
        assert ep.steps[0].content == "What is 2+2?"
        assert ep.steps[1].agent_name == "assistant"
        assert ep.steps[1].content == "4"
        assert ep.final_answer == "4"


class TestDSPyMetricBridge:
    def test_evaluate(self):
        bridge = DSPyMetricBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        score = bridge.evaluate("What is 2+2?", "4")
        assert score == 0.8

    def test_call_with_dicts(self):
        bridge = DSPyMetricBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        example = {"question": "What is 2+2?"}
        prediction = {"answer": "4"}
        result = bridge(example, prediction)
        assert result == 0.8

    def test_call_with_objects(self):
        class FakeExample:
            question = "test question"

        class FakePrediction:
            answer = "test answer"

        bridge = DSPyMetricBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        result = bridge(FakeExample(), FakePrediction())
        assert result == 0.8

    def test_threshold_mode_true(self):
        bridge = DSPyMetricBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
            threshold=0.5,
        )
        example = {"question": "q"}
        prediction = {"answer": "a"}
        result = bridge(example, prediction)
        assert result is True

    def test_threshold_mode_false(self):
        bridge = DSPyMetricBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
            threshold=0.9,
        )
        example = {"question": "q"}
        prediction = {"answer": "a"}
        result = bridge(example, prediction)
        assert result is False

    def test_custom_fields(self):
        bridge = DSPyMetricBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
            input_field="prompt",
            output_field="response",
        )
        example = {"prompt": "test"}
        prediction = {"response": "output"}
        result = bridge(example, prediction)
        assert result == 0.8

    def test_as_dspy_metric(self):
        bridge = DSPyMetricBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        metric = bridge.as_dspy_metric()
        assert metric is bridge

    def test_custom_episode_builder(self):
        def custom_builder(inp, out):
            return Episode(
                episode_id="custom",
                steps=[
                    Step(
                        kind=StepKind.MESSAGE, agent_id="a", agent_name="A", content=out
                    )
                ],
                source_framework="custom",
            )

        bridge = DSPyMetricBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
            episode_builder=custom_builder,
        )
        result = bridge.evaluate("in", "out")
        assert result == 0.8


class TestDSPyAvailability:
    def test_availability_flag(self):
        from agent_eval.rl.dspy_bridge import _DSPY_AVAILABLE

        assert isinstance(_DSPY_AVAILABLE, bool)
