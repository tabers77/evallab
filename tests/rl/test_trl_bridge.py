"""Tests for rl.trl_bridge — GRPORewardBridge."""

import pytest

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Issue, ScoreDimension, ScoreVector, Severity
from agent_eval.rl.trl_bridge import (
    GRPORewardBridge,
    _default_episode_builder,
    _tool_aware_episode_builder,
)


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

    def test_conversational_format_prompts(self):
        """Message-dict prompts/completions are handled correctly."""
        prompt = [{"role": "user", "content": "What is 2+2?"}]
        completion = [{"role": "assistant", "content": "4"}]
        ep = _default_episode_builder(prompt, completion)
        assert len(ep.steps) == 2
        assert ep.steps[0].content == "What is 2+2?"
        assert ep.steps[1].content == "4"
        assert ep.final_answer == "4"

    def test_conversational_format_multi_turn(self):
        """Multi-turn prompt extracts last message content."""
        prompt = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        completion = [
            {"role": "assistant", "content": "First reply"},
            {"role": "assistant", "content": "Second reply"},
        ]
        ep = _default_episode_builder(prompt, completion)
        assert ep.steps[0].content == "Hello"
        assert ep.steps[1].content == "Second reply"
        assert ep.final_answer == "Second reply"

    def test_empty_message_list(self):
        """Empty message lists produce empty content."""
        ep = _default_episode_builder([], [])
        assert ep.steps[0].content == ""
        assert ep.steps[1].content == ""


class TestToolAwareEpisodeBuilder:
    def test_plain_strings_fallback(self):
        """Plain strings delegate to the default builder."""
        ep = _tool_aware_episode_builder("hello", "world")
        assert len(ep.steps) == 2
        assert ep.steps[0].content == "hello"
        assert ep.steps[1].content == "world"

    def test_tool_call_messages(self):
        """Completion with tool_calls generates TOOL_CALL steps."""
        prompt = [{"role": "user", "content": "What's the weather?"}]
        completion = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "content": '{"temp": 72}',
            },
            {
                "role": "assistant",
                "content": "It's 72°F in NYC.",
            },
        ]
        ep = _tool_aware_episode_builder(prompt, completion)

        # prompt step + tool_call + tool_result + final assistant
        assert len(ep.steps) == 4
        assert ep.steps[0].kind == StepKind.MESSAGE
        assert ep.steps[0].content == "What's the weather?"
        assert ep.steps[1].kind == StepKind.TOOL_CALL
        assert "get_weather" in ep.steps[1].content
        assert ep.steps[2].kind == StepKind.TOOL_RESULT
        assert ep.steps[2].agent_name == "get_weather"
        assert ep.steps[3].kind == StepKind.MESSAGE
        assert ep.final_answer == "It's 72°F in NYC."

    def test_multiple_tool_calls(self):
        """Multiple tool calls in a single message produce multiple steps."""
        prompt = [{"role": "user", "content": "Compare weather"}]
        completion = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}},
                    {"id": "c2", "function": {"name": "get_weather", "arguments": '{"city": "LA"}'}},
                ],
            },
        ]
        ep = _tool_aware_episode_builder(prompt, completion)
        tool_call_steps = [s for s in ep.steps if s.kind == StepKind.TOOL_CALL]
        assert len(tool_call_steps) == 2


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

    def test_conversational_format_batch(self):
        """Batch with conversational format works end-to-end."""
        bridge = GRPORewardBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        rewards = bridge.compute_rewards(
            prompts=[
                [{"role": "user", "content": "q1"}],
                [{"role": "user", "content": "q2"}],
            ],
            completions=[
                [{"role": "assistant", "content": "a1"}],
                [{"role": "assistant", "content": "a2"}],
            ],
        )
        assert len(rewards) == 2
        assert all(r == 0.8 for r in rewards)

    def test_conversational_format_mixed(self):
        """Mix of string and message-dict in a batch works."""
        bridge = GRPORewardBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        rewards = bridge.compute_rewards(
            prompts=["plain question", [{"role": "user", "content": "dict question"}]],
            completions=["plain answer", [{"role": "assistant", "content": "dict answer"}]],
        )
        assert len(rewards) == 2
        assert all(r == 0.8 for r in rewards)

    def test_tools_parameter_activates_tool_builder(self):
        """When tools= is provided, the tool-aware builder is used automatically."""
        def dummy_tool(x: str) -> str:
            return x

        bridge = GRPORewardBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
            tools=[dummy_tool],
        )
        assert bridge.episode_builder is _tool_aware_episode_builder
        assert bridge.tools == [dummy_tool]

    def test_tools_with_custom_builder_keeps_custom(self):
        """Custom episode_builder takes precedence even when tools are given."""
        def custom_builder(prompt, completion):
            return _default_episode_builder(prompt, completion)

        def dummy_tool(x: str) -> str:
            return x

        bridge = GRPORewardBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
            episode_builder=custom_builder,
            tools=[dummy_tool],
        )
        assert bridge.episode_builder is custom_builder

    def test_as_trl_trainer_kwargs_with_tools(self):
        """as_trl_trainer_kwargs includes tools when configured."""
        def dummy_tool(x: str) -> str:
            return x

        bridge = GRPORewardBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
            tools=[dummy_tool],
        )
        kwargs = bridge.as_trl_trainer_kwargs()
        assert "reward_funcs" in kwargs
        assert len(kwargs["reward_funcs"]) == 1
        assert "tools" in kwargs
        assert kwargs["tools"] == [dummy_tool]

    def test_as_trl_trainer_kwargs_without_tools(self):
        """as_trl_trainer_kwargs omits tools key when none configured."""
        bridge = GRPORewardBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        kwargs = bridge.as_trl_trainer_kwargs()
        assert "reward_funcs" in kwargs
        assert "tools" not in kwargs

    def test_kwargs_passthrough(self):
        """Extra kwargs (completion_ids, trainer_state, etc.) don't break compute_rewards."""
        bridge = GRPORewardBridge(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        rewards = bridge.compute_rewards(
            prompts=["q1"],
            completions=["a1"],
            completion_ids=[[1, 2, 3]],
            trainer_state={"step": 42},
            custom_column=["extra_data"],
        )
        assert len(rewards) == 1
        assert rewards[0] == 0.8
