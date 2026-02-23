"""Tests for adapters.otel.adapter — OTelTraceAdapter."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from agent_eval.adapters.otel.adapter import OTelTraceAdapter
from agent_eval.core.exceptions import AdapterError
from agent_eval.core.models import StepKind


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
SAMPLE_TRACE = FIXTURES_DIR / "sample_otel_trace.json"


class TestOTelAdapterInit:
    def test_framework_name(self):
        adapter = OTelTraceAdapter()
        assert adapter.framework_name == "opentelemetry"

    def test_default_agent_name(self):
        adapter = OTelTraceAdapter()
        assert adapter.default_agent_name == "agent"

    def test_custom_agent_name(self):
        adapter = OTelTraceAdapter(default_agent_name="my_agent")
        assert adapter.default_agent_name == "my_agent"


class TestOTelAdapterLoadEpisode:
    def test_load_from_fixture(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        assert ep.source_framework == "opentelemetry"
        assert ep.episode_id == "abc123def456"
        assert len(ep.steps) == 7

    def test_step_kinds_present(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        kinds = {s.kind for s in ep.steps}
        assert StepKind.MESSAGE in kinds
        assert StepKind.LLM_CALL in kinds
        assert StepKind.TOOL_CALL in kinds

    def test_agent_names_resolved(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        agents = ep.agents
        assert "research_agent" in agents
        assert "writer_agent" in agents

    def test_agent_name_inherited_from_parent(self):
        """LLM calls under research_agent should inherit its name."""
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        llm_calls = ep.steps_by_kind(StepKind.LLM_CALL)
        # First two LLM calls are under research_agent root
        research_llm = [s for s in llm_calls if s.agent_name == "research_agent"]
        assert len(research_llm) >= 2

    def test_timestamps_extracted(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        assert ep.started_at is not None
        assert ep.ended_at is not None
        assert ep.ended_at > ep.started_at

    def test_metadata_includes_trace_id(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        assert ep.metadata.get("trace_id") == "abc123def456"

    def test_metadata_includes_service_name(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        assert ep.metadata.get("service_name") == "multi-agent-app"

    def test_metadata_includes_scope_name(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        assert ep.metadata.get("scope_name") == "opentelemetry.instrumentation.openai"

    def test_tool_calls_detected(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        tool_calls = ep.steps_by_kind(StepKind.TOOL_CALL)
        tool_names = {s.tool_name for s in tool_calls}
        assert "web_search" in tool_names
        assert "save_document" in tool_names

    def test_tool_success_and_failure(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        tool_calls = ep.steps_by_kind(StepKind.TOOL_CALL)
        web_search = [s for s in tool_calls if s.tool_name == "web_search"]
        save_doc = [s for s in tool_calls if s.tool_name == "save_document"]
        assert web_search[0].tool_succeeded is True
        assert save_doc[0].tool_succeeded is False

    def test_llm_calls_have_model(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        llm_calls = ep.steps_by_kind(StepKind.LLM_CALL)
        assert all(s.model == "gpt-4o" for s in llm_calls)

    def test_llm_calls_have_tokens(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        llm_calls = ep.steps_by_kind(StepKind.LLM_CALL)
        with_tokens = [s for s in llm_calls if s.prompt_tokens is not None]
        assert len(with_tokens) >= 1
        assert with_tokens[0].prompt_tokens > 0
        assert with_tokens[0].completion_tokens > 0

    def test_steps_ordered_chronologically(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        timestamps = [s.timestamp for s in ep.steps if s.timestamp]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_missing_file_raises(self):
        adapter = OTelTraceAdapter()
        with pytest.raises(AdapterError, match="not found"):
            adapter.load_episode("/nonexistent/trace.json")


class TestOTelAdapterLoadFromDict:
    def test_load_from_dict(self):
        adapter = OTelTraceAdapter()
        trace = json.loads(SAMPLE_TRACE.read_text())
        ep = adapter.load_from_dict(trace)
        assert ep.source_framework == "opentelemetry"
        assert len(ep.steps) == 7

    def test_empty_trace_raises(self):
        adapter = OTelTraceAdapter()
        with pytest.raises(AdapterError, match="no spans"):
            adapter.load_from_dict({"resourceSpans": []})

    def test_flat_spans_format(self):
        """Simplified {"spans": [...]} format should work."""
        adapter = OTelTraceAdapter()
        trace = {
            "spans": [
                {
                    "traceId": "test-trace",
                    "spanId": "s1",
                    "name": "chat gpt-4",
                    "startTimeUnixNano": "1700000000000000000",
                    "endTimeUnixNano": "1700000001000000000",
                    "attributes": [
                        {
                            "key": "gen_ai.operation.name",
                            "value": {"stringValue": "chat"},
                        }
                    ],
                }
            ]
        }
        ep = adapter.load_from_dict(trace)
        assert len(ep.steps) == 1
        assert ep.steps[0].kind == StepKind.LLM_CALL


class TestOTelAdapterLoadEpisodes:
    def test_load_from_directory(self):
        adapter = OTelTraceAdapter()
        episodes = adapter.load_episodes(str(FIXTURES_DIR))
        otel_episodes = [e for e in episodes if e.source_framework == "opentelemetry"]
        assert len(otel_episodes) >= 1

    def test_single_file_fallback(self):
        adapter = OTelTraceAdapter()
        episodes = adapter.load_episodes(str(SAMPLE_TRACE))
        assert len(episodes) == 1

    def test_jsonl_loading(self):
        """JSONL format: one OTLP JSON per line."""
        adapter = OTelTraceAdapter()
        trace = json.loads(SAMPLE_TRACE.read_text())
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps(trace) + "\n")
            f.write(json.dumps(trace) + "\n")
            tmppath = f.name

        try:
            ep = adapter.load_episode(tmppath)
            # Two copies of the trace merged — should have 14 spans
            assert len(ep.steps) == 14
        finally:
            os.unlink(tmppath)


class TestOTelAdapterEdgeCases:
    def test_span_without_attributes(self):
        adapter = OTelTraceAdapter()
        trace = {
            "spans": [
                {
                    "traceId": "t1",
                    "spanId": "s1",
                    "name": "unknown_operation",
                    "startTimeUnixNano": "1700000000000000000",
                    "endTimeUnixNano": "1700000001000000000",
                }
            ]
        }
        ep = adapter.load_from_dict(trace)
        assert len(ep.steps) == 1
        assert ep.steps[0].kind == StepKind.CUSTOM

    def test_span_with_empty_attributes(self):
        adapter = OTelTraceAdapter()
        trace = {
            "spans": [
                {
                    "traceId": "t1",
                    "spanId": "s1",
                    "name": "test",
                    "startTimeUnixNano": "1700000000000000000",
                    "endTimeUnixNano": "1700000001000000000",
                    "attributes": [],
                }
            ]
        }
        ep = adapter.load_from_dict(trace)
        assert len(ep.steps) == 1

    def test_default_agent_name_used(self):
        adapter = OTelTraceAdapter(default_agent_name="fallback_agent")
        trace = {
            "spans": [
                {
                    "traceId": "t1",
                    "spanId": "s1",
                    "name": "chat",
                    "startTimeUnixNano": "1700000000000000000",
                    "endTimeUnixNano": "1700000001000000000",
                    "attributes": [
                        {
                            "key": "gen_ai.operation.name",
                            "value": {"stringValue": "chat"},
                        }
                    ],
                }
            ]
        }
        ep = adapter.load_from_dict(trace)
        assert ep.steps[0].agent_name == "fallback_agent"

    def test_legacy_llm_attributes(self):
        """OpenLLMetry legacy attributes (llm.*) should be picked up."""
        adapter = OTelTraceAdapter()
        trace = {
            "spans": [
                {
                    "traceId": "t1",
                    "spanId": "s1",
                    "name": "llm.chat",
                    "startTimeUnixNano": "1700000000000000000",
                    "endTimeUnixNano": "1700000001000000000",
                    "attributes": [
                        {
                            "key": "llm.request.model",
                            "value": {"stringValue": "claude-3-opus"},
                        },
                        {
                            "key": "llm.usage.prompt_tokens",
                            "value": {"intValue": "100"},
                        },
                        {
                            "key": "llm.usage.completion_tokens",
                            "value": {"intValue": "50"},
                        },
                    ],
                }
            ]
        }
        ep = adapter.load_from_dict(trace)
        step = ep.steps[0]
        # Span name contains "llm" → LLM_CALL via fallback
        assert step.kind == StepKind.LLM_CALL
        assert step.model == "claude-3-opus"
        assert step.prompt_tokens == 100
        assert step.completion_tokens == 50

    def test_malformed_timestamp_handled(self):
        adapter = OTelTraceAdapter()
        trace = {
            "spans": [
                {
                    "traceId": "t1",
                    "spanId": "s1",
                    "name": "test",
                    "startTimeUnixNano": "not_a_number",
                    "endTimeUnixNano": "also_not_a_number",
                }
            ]
        }
        ep = adapter.load_from_dict(trace)
        assert len(ep.steps) == 1
        assert ep.steps[0].timestamp is None

    def test_content_extracted_from_completion(self):
        adapter = OTelTraceAdapter()
        ep = adapter.load_episode(str(SAMPLE_TRACE))
        llm_calls = ep.steps_by_kind(StepKind.LLM_CALL)
        with_content = [s for s in llm_calls if s.content]
        assert len(with_content) >= 1
        assert "renewable energy" in with_content[0].content.lower()
