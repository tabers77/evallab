"""Tests for rl.trl_bridge — GRPORewardBridge."""

import pytest

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Issue, ScoreDimension, ScoreVector, Severity
from agent_eval.rl.trl_bridge import GRPORewardBridge, _default_episode_builder


class FakeScorer:
    """A scorer that gives full marks for any episode."""

    @property
    def name(self):
        return "fake"

    def score(self, episode):
        return [ScoreDimension(name="quality", value=0.8, max_value=1.0)]

    def detect_issues(self, episode):
        return []


class IssueScorer:
    """A scorer that detects issues based on completion content."""

    @property
    def name(self):
        return "issue_scorer"

    def score(self, episode):
        return [ScoreDimension(name="quality", value=0.5, max_value=1.0)]

    def detect_issues(self, episode):
        issues = []
        for step in episode.steps:
            if step.content and "error" in step.content.lower():
                issues.append(
                    Issue(Severity.ERROR, "Content", "Error detected in output")
                )
        return issues


class FakeRewardFn:
    def compute(self, score_vector):
        return score_vector.overall


class TestDefaultEpisodeBuilder:
    def test_builds_two_step_episode(self):
        ep = _default_episode_builder("hello", "world")
        assert ep.source_framework == "trl"
        assert len(ep.steps) == 2
        assert ep.steps[0].kind == StepKind.MESSAGE
        assert ep.steps[0].agent_name == "user"
        assert ep.steps[0].content == "hello"
        assert ep.steps[1].agent_name == "assistant"
        assert ep.steps[1].content == "world"
        assert ep.final_answer == "world"


class TestGRPORewardBridge:
    def test_single_reward(self):
        bridge = GRPORewardBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        reward = bridge.compute_reward("What is 2+2?", "4")
        assert reward == 0.8

    def test_batch_rewards(self):
        bridge = GRPORewardBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        rewards = bridge.compute_rewards(
            prompts=["q1", "q2", "q3"],
            completions=["a1", "a2", "a3"],
        )
        assert len(rewards) == 3
        assert all(r == 0.8 for r in rewards)

    def test_mismatched_lengths_raises(self):
        bridge = GRPORewardBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        with pytest.raises(ValueError, match="same length"):
            bridge.compute_rewards(
                prompts=["q1", "q2"],
                completions=["a1"],
            )

    def test_as_trl_reward_fn(self):
        bridge = GRPORewardBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        fn = bridge.as_trl_reward_fn()
        result = fn(prompts=["q"], completions=["a"])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_with_issue_scorer(self):
        bridge = GRPORewardBridge(
            scorers=[IssueScorer()],
            reward_fn=FakeRewardFn(),
        )
        # Clean completion
        r1 = bridge.compute_reward("q", "good answer")
        # Completion with error
        r2 = bridge.compute_reward("q", "this has an error in it")
        # Both should return 0.5 (based on score dimension, not issues)
        assert r1 == 0.5
        assert r2 == 0.5

    def test_custom_episode_builder(self):
        def custom_builder(prompt, completion):
            return Episode(
                episode_id="custom",
                steps=[
                    Step(
                        kind=StepKind.LLM_CALL,
                        agent_id="agent",
                        agent_name="CustomAgent",
                        content=completion,
                    )
                ],
                source_framework="custom",
            )

        bridge = GRPORewardBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
            episode_builder=custom_builder,
        )
        reward = bridge.compute_reward("q", "a")
        assert reward == 0.8

    def test_multiple_scorers(self):
        bridge = GRPORewardBridge(
            scorers=[FakeScorer(), IssueScorer()],
            reward_fn=FakeRewardFn(),
        )
        reward = bridge.compute_reward("q", "answer")
        # Two dimensions: quality=0.8 and quality=0.5 -> avg = 0.65
        assert abs(reward - 0.65) < 1e-9
