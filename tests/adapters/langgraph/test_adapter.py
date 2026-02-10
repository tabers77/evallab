"""Tests for adapters.langgraph.adapter — LangGraphAdapter."""

import json
from pathlib import Path

import pytest

from agent_eval.adapters.langgraph.adapter import LangGraphAdapter
from agent_eval.core.exceptions import AdapterError
from agent_eval.core.models import StepKind


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
SAMPLE_TRACE = FIXTURES_DIR / "sample_langgraph_trace.json"


class TestLangGraphAdapterInit:
    def test_framework_name(self):
        adapter = LangGraphAdapter()
        assert adapter.framework_name == "langgraph"

    def test_custom_agent_name(self):
        adapter = LangGraphAdapter(default_agent_name="my_agent")
        assert adapter.default_agent_name == "my_agent"


class TestLangGraphAdapterLoadEpisode:
    def test_load_from_fixture(self):
        adapter = LangGraphAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        assert ep.source_framework == "langgraph"
        assert ep.episode_id == "test-thread-001"
        assert len(ep.steps) > 0

    def test_step_kinds_present(self):
        adapter = LangGraphAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        kinds = {s.kind for s in ep.steps}
        assert StepKind.MESSAGE in kinds
        assert StepKind.LLM_CALL in kinds
        assert StepKind.TOOL_CALL in kinds
        assert StepKind.TOOL_RESULT in kinds

    def test_tool_calls_detected(self):
        adapter = LangGraphAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        tool_calls = ep.steps_by_kind(StepKind.TOOL_CALL)
        tool_names = {s.tool_name for s in tool_calls}
        assert "get_revenue" in tool_names
        assert "plot_bar_chart" in tool_names

    def test_tool_results_detected(self):
        adapter = LangGraphAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        tool_results = ep.steps_by_kind(StepKind.TOOL_RESULT)
        assert len(tool_results) >= 2
        # get_revenue result should be successful
        revenue_results = [s for s in tool_results if s.tool_name == "get_revenue"]
        assert len(revenue_results) == 1
        assert revenue_results[0].tool_succeeded is True

    def test_llm_calls_detected(self):
        adapter = LangGraphAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        llm_calls = ep.steps_by_kind(StepKind.LLM_CALL)
        assert len(llm_calls) >= 2
        models = {s.model for s in llm_calls if s.model}
        assert "gpt-4o" in models

    def test_token_counts(self):
        adapter = LangGraphAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        llm_calls = ep.steps_by_kind(StepKind.LLM_CALL)
        # on_chat_model_end events should have token counts
        with_tokens = [s for s in llm_calls if s.prompt_tokens is not None]
        assert len(with_tokens) >= 1
        assert with_tokens[0].prompt_tokens > 0

    def test_timestamps_extracted(self):
        adapter = LangGraphAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        assert ep.started_at is not None
        assert ep.ended_at is not None
        assert ep.ended_at > ep.started_at

    def test_task_description_extracted(self):
        adapter = LangGraphAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        assert ep.task_description is not None
        assert "revenue" in ep.task_description.lower()

    def test_final_answer_extracted(self):
        adapter = LangGraphAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        assert ep.final_answer is not None
        assert "283M" in ep.final_answer or "283" in ep.final_answer

    def test_metadata_includes_thread_id(self):
        adapter = LangGraphAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        assert ep.metadata.get("thread_id") == "test-thread-001"

    def test_missing_file_raises(self):
        adapter = LangGraphAdapter()
        with pytest.raises(AdapterError, match="not found"):
            adapter.load_episode("/nonexistent/trace.json")


class TestLangGraphAdapterLoadFromDict:
    def test_load_from_dict(self):
        adapter = LangGraphAdapter()
        trace = json.loads(SAMPLE_TRACE.read_text())
        ep = adapter.load_from_dict(trace)
        assert ep.source_framework == "langgraph"
        assert len(ep.steps) > 0

    def test_empty_events_raises(self):
        adapter = LangGraphAdapter()
        with pytest.raises(AdapterError, match="events"):
            adapter.load_from_dict({"events": []})


class TestLangGraphAdapterLoadEpisodes:
    def test_load_from_directory(self):
        adapter = LangGraphAdapter()
        episodes = adapter.load_episodes(str(FIXTURES_DIR))
        # Should find the sample_langgraph_trace.json
        lg_episodes = [e for e in episodes if e.source_framework == "langgraph"]
        assert len(lg_episodes) >= 1

    def test_single_file_fallback(self):
        adapter = LangGraphAdapter()
        episodes = adapter.load_episodes(str(SAMPLE_TRACE))
        assert len(episodes) == 1


class TestLangGraphAdapterErrorDetection:
    def test_tool_error_detected(self):
        adapter = LangGraphAdapter()
        trace = {
            "thread_id": "err-test",
            "events": [
                {
                    "event": "on_chain_start",
                    "name": "agent",
                    "data": {"input": "test"},
                },
                {
                    "event": "on_tool_start",
                    "name": "bad_tool",
                    "data": {"input": {}},
                },
                {
                    "event": "on_tool_end",
                    "name": "bad_tool",
                    "data": {"output": {"error": "Connection refused"}},
                },
                {
                    "event": "on_chain_end",
                    "name": "agent",
                    "data": {"output": "Failed to retrieve data"},
                },
            ],
        }
        ep = adapter.load_from_dict(trace)
        tool_results = ep.steps_by_kind(StepKind.TOOL_RESULT)
        assert len(tool_results) == 1
        assert tool_results[0].tool_succeeded is False
