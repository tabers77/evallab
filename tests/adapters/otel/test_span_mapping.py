"""Tests for adapters.otel.span_mapping — span→step mapping logic."""

import pytest

from agent_eval.adapters.otel.span_mapping import (
    get_attr,
    infer_step_kind,
    span_to_step,
    unwrap_attribute_value,
)
from agent_eval.core.models import StepKind


class TestUnwrapAttributeValue:
    def test_string_value(self):
        assert unwrap_attribute_value({"stringValue": "hello"}) == "hello"

    def test_int_value_as_string(self):
        assert unwrap_attribute_value({"intValue": "42"}) == 42

    def test_int_value_as_int(self):
        assert unwrap_attribute_value({"intValue": 42}) == 42

    def test_double_value(self):
        assert unwrap_attribute_value({"doubleValue": 3.14}) == 3.14

    def test_bool_value(self):
        assert unwrap_attribute_value({"boolValue": True}) is True
        assert unwrap_attribute_value({"boolValue": False}) is False

    def test_array_value(self):
        result = unwrap_attribute_value(
            {
                "arrayValue": {
                    "values": [
                        {"stringValue": "a"},
                        {"intValue": "1"},
                    ]
                }
            }
        )
        assert result == ["a", 1]

    def test_array_value_empty(self):
        result = unwrap_attribute_value({"arrayValue": {"values": []}})
        assert result == []

    def test_kvlist_value(self):
        result = unwrap_attribute_value(
            {
                "kvlistValue": {
                    "values": [
                        {"key": "name", "value": {"stringValue": "test"}},
                        {"key": "count", "value": {"intValue": "5"}},
                    ]
                }
            }
        )
        assert result == {"name": "test", "count": 5}

    def test_plain_value_passthrough(self):
        assert unwrap_attribute_value("plain") == "plain"
        assert unwrap_attribute_value(42) == 42

    def test_unknown_dict_passthrough(self):
        val = {"unknownType": "something"}
        assert unwrap_attribute_value(val) == val


class TestGetAttr:
    def test_found(self):
        span = {
            "attributes": [
                {"key": "gen_ai.request.model", "value": {"stringValue": "gpt-4o"}}
            ]
        }
        assert get_attr(span, "gen_ai.request.model") == "gpt-4o"

    def test_not_found(self):
        span = {"attributes": []}
        assert get_attr(span, "gen_ai.request.model") is None

    def test_no_attributes_key(self):
        span = {}
        assert get_attr(span, "gen_ai.request.model") is None


class TestInferStepKind:
    @pytest.mark.parametrize(
        "operation,expected",
        [
            ("chat", StepKind.LLM_CALL),
            ("text_completion", StepKind.LLM_CALL),
            ("generate_content", StepKind.LLM_CALL),
            ("execute_tool", StepKind.TOOL_CALL),
            ("invoke_agent", StepKind.MESSAGE),
            ("create_agent", StepKind.MESSAGE),
            ("embeddings", StepKind.CUSTOM),
            ("retrieval", StepKind.CUSTOM),
        ],
    )
    def test_operation_name_mapping(self, operation, expected):
        span = {
            "name": "test",
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": operation}}
            ],
        }
        assert infer_step_kind(span) == expected

    def test_fallback_to_span_name_llm(self):
        span = {"name": "llm.chat.completion", "attributes": []}
        assert infer_step_kind(span) == StepKind.LLM_CALL

    def test_fallback_to_span_name_tool(self):
        span = {"name": "tool_execution", "attributes": []}
        assert infer_step_kind(span) == StepKind.TOOL_CALL

    def test_fallback_to_span_name_agent(self):
        span = {"name": "agent_invocation", "attributes": []}
        assert infer_step_kind(span) == StepKind.MESSAGE

    def test_unknown_defaults_to_custom(self):
        span = {"name": "some_random_span", "attributes": []}
        assert infer_step_kind(span) == StepKind.CUSTOM


class TestSpanToStep:
    def test_llm_call_step(self):
        span = {
            "spanId": "s1",
            "name": "chat",
            "startTimeUnixNano": "1700000000000000000",
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.request.model", "value": {"stringValue": "gpt-4o"}},
                {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "100"}},
                {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "50"}},
            ],
        }
        step = span_to_step(span, None, "default")
        assert step.kind == StepKind.LLM_CALL
        assert step.model == "gpt-4o"
        assert step.prompt_tokens == 100
        assert step.completion_tokens == 50
        assert step.agent_name == "default"

    def test_tool_call_step(self):
        span = {
            "spanId": "s2",
            "name": "execute_tool",
            "startTimeUnixNano": "1700000000000000000",
            "status": {"code": 1},
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": "calculator"}},
                {"key": "gen_ai.tool.call.result", "value": {"stringValue": "42"}},
            ],
        }
        step = span_to_step(span, None, "default")
        assert step.kind == StepKind.TOOL_CALL
        assert step.tool_name == "calculator"
        assert step.tool_result == "42"
        assert step.tool_succeeded is True

    def test_tool_failure_from_status(self):
        span = {
            "spanId": "s3",
            "name": "execute_tool",
            "startTimeUnixNano": "1700000000000000000",
            "status": {"code": 2, "message": "error"},
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": "broken_tool"}},
            ],
        }
        step = span_to_step(span, None, "default")
        assert step.tool_succeeded is False

    def test_agent_name_from_span(self):
        span = {
            "spanId": "s4",
            "name": "invoke",
            "startTimeUnixNano": "1700000000000000000",
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "invoke_agent"}},
                {"key": "gen_ai.agent.name", "value": {"stringValue": "planner"}},
            ],
        }
        step = span_to_step(span, None, "default")
        assert step.agent_name == "planner"

    def test_agent_name_inherited_from_parent(self):
        parent = {
            "spanId": "p1",
            "name": "agent_root",
            "attributes": [
                {"key": "gen_ai.agent.name", "value": {"stringValue": "research_agent"}},
            ],
        }
        child = {
            "spanId": "c1",
            "name": "chat",
            "startTimeUnixNano": "1700000000000000000",
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            ],
        }
        step = span_to_step(child, parent, "default")
        assert step.agent_name == "research_agent"

    def test_legacy_traceloop_tool_name(self):
        span = {
            "spanId": "s5",
            "name": "tool_call",
            "startTimeUnixNano": "1700000000000000000",
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                {"key": "traceloop.entity.name", "value": {"stringValue": "search_api"}},
            ],
        }
        step = span_to_step(span, None, "default")
        assert step.tool_name == "search_api"

    def test_provider_in_metadata(self):
        span = {
            "spanId": "s6",
            "name": "chat",
            "startTimeUnixNano": "1700000000000000000",
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
            ],
        }
        step = span_to_step(span, None, "default")
        assert step.metadata.get("provider") == "openai"

    def test_timestamp_parsed(self):
        span = {
            "spanId": "s7",
            "name": "test",
            "startTimeUnixNano": "1700000000000000000",
        }
        step = span_to_step(span, None, "default")
        assert step.timestamp is not None
        assert step.timestamp.year == 2023

    def test_content_from_completion(self):
        span = {
            "spanId": "s8",
            "name": "chat",
            "startTimeUnixNano": "1700000000000000000",
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.content.completion", "value": {"stringValue": "Hello world"}},
            ],
        }
        step = span_to_step(span, None, "default")
        assert step.content == "Hello world"
