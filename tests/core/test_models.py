"""Tests for core.models — Episode, Step, StepKind."""

from datetime import datetime, timezone

from agent_eval.core.models import Episode, Step, StepKind


class TestStepKind:
    def test_enum_values(self):
        assert StepKind.MESSAGE.value == "message"
        assert StepKind.TOOL_CALL.value == "tool_call"
        assert StepKind.LLM_CALL.value == "llm_call"
        assert StepKind.FACT_CHECK.value == "fact_check"
        assert StepKind.CUSTOM.value == "custom"

    def test_string_comparison(self):
        assert StepKind.MESSAGE == "message"
        assert StepKind.TOOL_CALL == "tool_call"


class TestStep:
    def test_minimal_step(self):
        step = Step(kind=StepKind.MESSAGE, agent_id="a1", agent_name="Agent1")
        assert step.kind == StepKind.MESSAGE
        assert step.agent_id == "a1"
        assert step.content is None
        assert step.metadata == {}

    def test_tool_call_step(self):
        step = Step(
            kind=StepKind.TOOL_CALL,
            agent_id="a1",
            agent_name="Agent1",
            tool_name="get_data",
            tool_args={"key": "value"},
            tool_result={"data": 42},
            tool_succeeded=True,
        )
        assert step.tool_name == "get_data"
        assert step.tool_succeeded is True

    def test_llm_call_step(self):
        step = Step(
            kind=StepKind.LLM_CALL,
            agent_id="a1",
            agent_name="Agent1",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert step.model == "gpt-4o"
        assert step.prompt_tokens == 100


class TestEpisode:
    def test_agents_property(self, minimal_episode):
        agents = minimal_episode.agents
        assert "Agent1" in agents
        # Tool call steps have empty agent_name, filtered out by .agents
        assert "" not in agents

    def test_duration_seconds(self, minimal_episode):
        assert minimal_episode.duration_seconds == 120.0

    def test_duration_none_without_timestamps(self):
        ep = Episode(
            episode_id="test",
            steps=[],
            source_framework="test",
        )
        assert ep.duration_seconds is None

    def test_steps_by_kind(self, minimal_episode):
        messages = minimal_episode.steps_by_kind(StepKind.MESSAGE)
        assert len(messages) == 1
        assert messages[0].content == "Analyzing data"

        tools = minimal_episode.steps_by_kind(StepKind.TOOL_CALL)
        assert len(tools) == 1
        assert tools[0].tool_name == "get_data"

    def test_steps_by_agent(self, minimal_episode):
        agent1_steps = minimal_episode.steps_by_agent("Agent1")
        assert len(agent1_steps) == 1

    def test_empty_episode(self):
        ep = Episode(
            episode_id="empty",
            steps=[],
            source_framework="test",
        )
        assert ep.agents == set()
        assert ep.duration_seconds is None
        assert ep.final_answer is None
