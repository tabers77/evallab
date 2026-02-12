"""Tests for adapters.autogen.adapter — AutoGenAdapter."""

import json
import tempfile
from pathlib import Path

import pytest

from agent_eval.adapters.autogen.adapter import AutoGenAdapter
from agent_eval.core.exceptions import AdapterError
from agent_eval.core.models import StepKind


AGENT_NAMES = [
    "SalesNegotiator",
    "FinanceExpert",
    "CustomerResearcher",
    "DataVisualiser",
]


class TestAutoGenAdapterInit:
    def test_framework_name(self):
        adapter = AutoGenAdapter()
        assert adapter.framework_name == "autogen"

    def test_custom_agent_names(self):
        adapter = AutoGenAdapter(agent_names=["Agent1", "Agent2"])
        assert adapter.agent_names == ["Agent1", "Agent2"]


class TestAutoGenAdapterLoadEpisode:
    def test_load_from_fixture(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        assert episode.source_framework == "autogen"
        assert episode.episode_id  # should be a UUID
        assert len(episode.steps) > 0

    def test_load_from_directory(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        # Pass the directory containing event.txt
        episode = adapter.load_episode(str(sample_log_path.parent))
        assert len(episode.steps) > 0

    def test_step_kinds_present(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        kinds = {s.kind for s in episode.steps}
        assert StepKind.MESSAGE in kinds
        assert StepKind.TOOL_CALL in kinds
        assert StepKind.LLM_CALL in kinds
        assert StepKind.FACT_CHECK in kinds

    def test_tool_calls_detected(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        tool_steps = episode.steps_by_kind(StepKind.TOOL_CALL)
        assert len(tool_steps) == 3
        tool_names = {s.tool_name for s in tool_steps}
        assert "get_product_finances" in tool_names
        assert "ask_web" in tool_names
        assert "plot_bar_chart" in tool_names

    def test_tool_success_detection(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        tool_steps = episode.steps_by_kind(StepKind.TOOL_CALL)
        assert all(s.tool_succeeded is True for s in tool_steps)

    def test_agent_names_resolved(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        msg_steps = episode.steps_by_kind(StepKind.MESSAGE)
        agent_names = {s.agent_name for s in msg_steps}
        assert "FinanceExpert" in agent_names
        assert "CustomerResearcher" in agent_names
        assert "DataVisualiser" in agent_names

    def test_timestamps_extracted(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        assert episode.started_at is not None
        assert episode.ended_at is not None
        assert episode.duration_seconds > 0

    def test_final_answer_extracted(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        assert episode.final_answer is not None
        assert "283M" in episode.final_answer

    def test_fact_check_steps(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        fc_steps = episode.steps_by_kind(StepKind.FACT_CHECK)
        assert len(fc_steps) == 1
        assert fc_steps[0].metadata["verdict"] == "PASS"

    def test_missing_path_raises(self):
        adapter = AutoGenAdapter()
        with pytest.raises(AdapterError, match="Could not find AutoGen log file"):
            adapter.load_episode("/nonexistent/path")


class TestAutoGenAdapterLoadEpisodes:
    def test_load_from_fixture_dir(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episodes = adapter.load_episodes(str(sample_log_path.parent))
        assert len(episodes) == 1

    def test_load_single_file_fallback(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episodes = adapter.load_episodes(str(sample_log_path))
        assert len(episodes) == 1

    def test_load_from_temp_dir_with_multiple_logs(self):
        adapter = AutoGenAdapter()

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                log_dir = Path(tmpdir) / f"log-{i}"
                log_dir.mkdir()
                (log_dir / "event.txt").write_text(
                    f'{{"type": "Message", "sender": "Agent{i}", "content": "msg"}}'
                )

            episodes = adapter.load_episodes(tmpdir)
            assert len(episodes) == 3


# ------------------------------------------------------------------
# New event type tests
# ------------------------------------------------------------------


class TestAutoGenAdapterNewEventTypes:
    """Tests for the 7 new event types added by the structured logger."""

    def _steps_from_events(self, events, agent_names=None):
        adapter = AutoGenAdapter(agent_names=agent_names or [])
        return adapter._events_to_steps(events)

    def test_message_event(self):
        event = {
            "type": "MessageEvent",
            "sender": "Agent1",
            "receiver": "Agent2",
            "payload": "Hello from structured logger",
            "kind": "text",
            "delivery_stage": "deliver",
            "timestamp": "2026-01-15T14:00:00",
        }
        steps = self._steps_from_events([event])
        assert len(steps) == 1
        s = steps[0]
        assert s.kind == StepKind.MESSAGE
        assert s.agent_id == "Agent1"
        assert s.content == "Hello from structured logger"
        assert s.metadata["receiver"] == "Agent2"
        assert s.metadata["kind"] == "text"
        assert s.metadata["delivery_stage"] == "deliver"

    def test_message_event_dict_payload(self):
        event = {
            "type": "MessageEvent",
            "sender": "Agent1",
            "payload": {"key": "value"},
        }
        steps = self._steps_from_events([event])
        assert len(steps) == 1
        assert steps[0].content == "{'key': 'value'}"

    def test_message_dropped_event(self):
        event = {
            "type": "MessageDroppedEvent",
            "sender": "Agent1",
            "receiver": "Agent2",
            "reason": "Queue full",
            "payload": "Some data",
            "timestamp": "2026-01-15T14:00:00",
        }
        steps = self._steps_from_events([event])
        assert len(steps) == 1
        s = steps[0]
        assert s.kind == StepKind.CUSTOM
        assert s.content == "Queue full"
        assert s.metadata["custom_type"] == "MessageDroppedEvent"
        assert s.metadata["receiver"] == "Agent2"
        assert s.metadata["reason"] == "Queue full"

    def test_message_handler_exception_event(self):
        event = {
            "type": "MessageHandlerExceptionEvent",
            "handler_class": "MyHandler",
            "exception": "ValueError: bad input",
            "traceback": "Traceback ...",
            "timestamp": "2026-01-15T14:00:00",
        }
        steps = self._steps_from_events([event])
        assert len(steps) == 1
        s = steps[0]
        assert s.kind == StepKind.CUSTOM
        assert s.agent_id == "MyHandler"
        assert s.content == "ValueError: bad input"
        assert s.metadata["custom_type"] == "MessageHandlerExceptionEvent"
        assert s.metadata["traceback"] == "Traceback ..."

    def test_tool_call_event(self):
        event = {
            "type": "ToolCallEvent",
            "tool_name": "search_db",
            "arguments": {"query": "revenue"},
            "result": "Found 5 records",
            "agent_id": "ResearchAgent",
            "timestamp": "2026-01-15T14:00:00",
        }
        steps = self._steps_from_events([event])
        assert len(steps) == 1
        s = steps[0]
        assert s.kind == StepKind.TOOL_CALL
        assert s.tool_name == "search_db"
        assert s.tool_args == {"query": "revenue"}
        assert s.tool_result == "Found 5 records"
        assert s.tool_succeeded is True
        assert s.agent_id == "ResearchAgent"

    def test_tool_call_event_with_failure(self):
        event = {
            "type": "ToolCallEvent",
            "tool_name": "bad_tool",
            "result": {"error": "Connection timeout"},
            "agent_id": "Agent1",
        }
        steps = self._steps_from_events([event])
        assert len(steps) == 1
        assert steps[0].tool_succeeded is False

    def test_llm_stream_start_event(self):
        event = {
            "type": "LLMStreamStartEvent",
            "model": "gpt-4o",
            "agent_id": "FinanceExpert_abc",
            "timestamp": "2026-01-15T14:00:00",
        }
        steps = self._steps_from_events(
            [event], agent_names=["FinanceExpert"]
        )
        assert len(steps) == 1
        s = steps[0]
        assert s.kind == StepKind.LLM_CALL
        assert s.model == "gpt-4o"
        assert s.agent_name == "FinanceExpert"
        assert s.metadata["stream_phase"] == "start"

    def test_llm_stream_end_event(self):
        event = {
            "type": "LLMStreamEndEvent",
            "model": "gpt-4o",
            "agent_id": "FinanceExpert_abc",
            "prompt_tokens": 2000,
            "completion_tokens": 500,
            "timestamp": "2026-01-15T14:00:00",
        }
        steps = self._steps_from_events(
            [event], agent_names=["FinanceExpert"]
        )
        assert len(steps) == 1
        s = steps[0]
        assert s.kind == StepKind.LLM_CALL
        assert s.prompt_tokens == 2000
        assert s.completion_tokens == 500
        assert s.metadata["stream_phase"] == "end"

    def test_agent_construction_exception_event(self):
        event = {
            "type": "AgentConstructionExceptionEvent",
            "agent_class": "SpecialistAgent",
            "exception": "ImportError: missing module",
            "timestamp": "2026-01-15T14:00:00",
        }
        steps = self._steps_from_events([event])
        assert len(steps) == 1
        s = steps[0]
        assert s.kind == StepKind.CUSTOM
        assert s.agent_id == "SpecialistAgent"
        assert s.content == "ImportError: missing module"
        assert s.metadata["custom_type"] == "AgentConstructionExceptionEvent"
        assert s.metadata["agent_class"] == "SpecialistAgent"

    def test_unknown_event_type_skipped(self):
        events = [
            {"type": "UnknownFutureEvent", "data": "xyz"},
            {"type": "Message", "sender": "A", "content": "ok"},
        ]
        steps = self._steps_from_events(events)
        assert len(steps) == 1
        assert steps[0].kind == StepKind.MESSAGE


# ------------------------------------------------------------------
# Multi-format loading tests
# ------------------------------------------------------------------


class TestAutoGenAdapterMultiFormat:
    """Tests for loading JSONL, JSON array, and backward-compatible text."""

    def test_load_jsonl_file(self, sample_jsonl_path):
        adapter = AutoGenAdapter(
            agent_names=["FinanceExpert", "CustomerResearcher", "DataVisualiser"]
        )
        episode = adapter.load_episode(str(sample_jsonl_path))
        assert episode.source_framework == "autogen"
        assert len(episode.steps) == 11
        kinds = {s.kind for s in episode.steps}
        assert StepKind.MESSAGE in kinds
        assert StepKind.TOOL_CALL in kinds
        assert StepKind.LLM_CALL in kinds
        assert StepKind.FACT_CHECK in kinds
        assert StepKind.CUSTOM in kinds

    def test_load_json_array_file(self, sample_json_array_path):
        adapter = AutoGenAdapter(
            agent_names=["FinanceExpert", "CustomerResearcher", "DataVisualiser"]
        )
        episode = adapter.load_episode(str(sample_json_array_path))
        assert len(episode.steps) == 11

    def test_jsonl_and_json_array_produce_same_steps(
        self, sample_jsonl_path, sample_json_array_path
    ):
        adapter = AutoGenAdapter(
            agent_names=["FinanceExpert", "CustomerResearcher", "DataVisualiser"]
        )
        ep_jsonl = adapter.load_episode(str(sample_jsonl_path))
        ep_json = adapter.load_episode(str(sample_json_array_path))
        assert len(ep_jsonl.steps) == len(ep_json.steps)
        for a, b in zip(ep_jsonl.steps, ep_json.steps):
            assert a.kind == b.kind
            assert a.agent_id == b.agent_id

    def test_load_text_backward_compat(self, sample_log_path):
        """Existing text-format loading still works."""
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))
        assert len(episode.steps) > 0
        kinds = {s.kind for s in episode.steps}
        assert StepKind.MESSAGE in kinds

    def test_resolve_log_path_prefers_event_txt(self):
        """When a directory has both event.txt and events.jsonl, prefer event.txt."""
        adapter = AutoGenAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "event.txt").write_text(
                '{"type": "Message", "sender": "A", "content": "from txt"}'
            )
            (Path(tmpdir) / "events.jsonl").write_text(
                '{"type": "Message", "sender": "B", "content": "from jsonl"}'
            )
            episode = adapter.load_episode(tmpdir)
            msg = episode.steps[0]
            assert msg.agent_id == "A"

    def test_resolve_log_path_falls_back_to_jsonl(self):
        adapter = AutoGenAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "events.jsonl").write_text(
                '{"type": "Message", "sender": "B", "content": "jsonl"}'
            )
            episode = adapter.load_episode(tmpdir)
            assert len(episode.steps) == 1
            assert episode.steps[0].agent_id == "B"

    def test_resolve_log_path_falls_back_to_json(self):
        adapter = AutoGenAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "events.json").write_text(
                json.dumps([
                    {"type": "Message", "sender": "C", "content": "json"}
                ])
            )
            episode = adapter.load_episode(tmpdir)
            assert len(episode.steps) == 1
            assert episode.steps[0].agent_id == "C"

    def test_load_episodes_discovers_jsonl(self):
        adapter = AutoGenAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = Path(tmpdir) / "run1"
            d1.mkdir()
            (d1 / "events.jsonl").write_text(
                '{"type": "Message", "sender": "A", "content": "m"}\n'
            )
            d2 = Path(tmpdir) / "run2"
            d2.mkdir()
            (d2 / "events.jsonl").write_text(
                '{"type": "Message", "sender": "B", "content": "m"}\n'
            )
            episodes = adapter.load_episodes(tmpdir)
            assert len(episodes) == 2

    def test_load_episodes_skips_empty(self):
        """Episodes with 0 steps are not returned."""
        adapter = AutoGenAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = Path(tmpdir) / "run1"
            d1.mkdir()
            (d1 / "events.jsonl").write_text(
                '{"type": "Message", "sender": "A", "content": "m"}\n'
            )
            d2 = Path(tmpdir) / "run2"
            d2.mkdir()
            # No type key → 0 steps
            (d2 / "events.json").write_text('[{"no_type": true}]')
            episodes = adapter.load_episodes(tmpdir)
            assert len(episodes) == 1
