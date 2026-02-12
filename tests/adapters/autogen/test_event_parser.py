"""Tests for adapters.autogen.event_parser — JSON event extraction."""

import json

from agent_eval.adapters.autogen.event_parser import (
    detect_format,
    extract_json_events,
    extract_json_events_json_array,
    extract_json_events_jsonl,
    parse_events,
)


class TestExtractJsonEvents:
    def test_single_event(self):
        content = """
2026-01-15 14:04:11,206 autogen_core.events
{
  "type": "Message",
  "sender": "FinanceExpert",
  "receiver": "Orchestrator",
  "content": "Analyzing data"
}
"""
        events = extract_json_events(content)
        assert len(events) == 1
        assert events[0]["type"] == "Message"
        assert events[0]["sender"] == "FinanceExpert"
        assert events[0]["content"] == "Analyzing data"

    def test_multiple_events(self):
        content = """
{
  "type": "Message",
  "sender": "Agent1",
  "content": "Message 1"
}
{
  "type": "ToolCall",
  "tool": "get_data",
  "result": "Success"
}
{
  "type": "LLMCall",
  "model": "gpt-4o"
}
"""
        events = extract_json_events(content)
        assert len(events) == 3
        assert events[0]["type"] == "Message"
        assert events[1]["type"] == "ToolCall"
        assert events[2]["type"] == "LLMCall"

    def test_nested_objects(self):
        content = """
{
  "type": "ToolCall",
  "tool": "get_finances",
  "arguments": {
    "customer": "Tetra Pak",
    "year": 2025
  },
  "result": {
    "revenue": 283399382.94,
    "status": "success"
  }
}
"""
        events = extract_json_events(content)
        assert len(events) == 1
        assert events[0]["arguments"]["customer"] == "Tetra Pak"
        assert events[0]["result"]["revenue"] == 283399382.94

    def test_skips_malformed_json(self):
        content = """
{
  "type": "Message",
  "bad": json syntax here
}
{
  "type": "ToolCall",
  "tool": "valid_tool"
}
"""
        events = extract_json_events(content)
        assert len(events) == 1
        assert events[0]["type"] == "ToolCall"

    def test_filters_objects_without_type(self):
        content = """
{
  "some_data": "value",
  "number": 123
}
{
  "type": "Message",
  "content": "Valid event"
}
"""
        events = extract_json_events(content)
        assert len(events) == 1
        assert events[0]["type"] == "Message"

    def test_empty_content(self):
        events = extract_json_events("")
        assert events == []

    def test_no_json(self):
        events = extract_json_events("Just plain text\nNo JSON here\n")
        assert events == []

    def test_fact_check_event(self):
        content = """
{
  "type": "FactCheckResult",
  "agent_name": "FinanceExpert",
  "verdict": "PASS",
  "reasoning": ["Step 1: Verified", "Step 2: Confirmed"],
  "question": "What is the revenue?",
  "answer_preview": "Revenue is 283M",
  "timestamp": "2026-01-19T10:00:00"
}
"""
        events = extract_json_events(content)
        assert len(events) == 1
        assert events[0]["type"] == "FactCheckResult"
        assert events[0]["verdict"] == "PASS"
        assert len(events[0]["reasoning"]) == 2


# ------------------------------------------------------------------
# detect_format
# ------------------------------------------------------------------


class TestDetectFormat:
    def test_json_array(self):
        content = '[{"type": "Message"}]'
        assert detect_format(content) == "json_array"

    def test_json_array_with_leading_whitespace(self):
        content = '  \n  [{"type": "Message"}]'
        assert detect_format(content) == "json_array"

    def test_jsonl(self):
        content = '{"type": "Message", "sender": "Agent1"}\n{"type": "ToolCall"}\n'
        assert detect_format(content) == "jsonl"

    def test_jsonl_with_leading_blank_lines(self):
        content = '\n\n{"type": "Message", "sender": "Agent1"}\n'
        assert detect_format(content) == "jsonl"

    def test_text_format(self):
        content = (
            "2026-01-15 14:04:11,206 autogen_core.events\n"
            '{\n  "type": "Message"\n}\n'
        )
        assert detect_format(content) == "text"

    def test_empty_string(self):
        assert detect_format("") == "text"

    def test_plain_text(self):
        assert detect_format("Just some log lines\nNo JSON\n") == "text"

    def test_malformed_first_line_falls_to_text(self):
        content = "{not valid json}\n"
        assert detect_format(content) == "text"


# ------------------------------------------------------------------
# extract_json_events_jsonl
# ------------------------------------------------------------------


class TestExtractJsonEventsJsonl:
    def test_basic(self):
        content = (
            '{"type": "Message", "sender": "A"}\n'
            '{"type": "ToolCall", "tool": "t"}\n'
        )
        events = extract_json_events_jsonl(content)
        assert len(events) == 2
        assert events[0]["type"] == "Message"
        assert events[1]["type"] == "ToolCall"

    def test_skips_blank_lines(self):
        content = '\n{"type": "Message"}\n\n{"type": "LLMCall"}\n\n'
        events = extract_json_events_jsonl(content)
        assert len(events) == 2

    def test_skips_malformed_lines(self):
        content = (
            '{"type": "Message"}\n'
            "not json\n"
            '{"type": "ToolCall", "tool": "x"}\n'
        )
        events = extract_json_events_jsonl(content)
        assert len(events) == 2

    def test_filters_objects_without_type(self):
        content = '{"data": "value"}\n{"type": "Message"}\n'
        events = extract_json_events_jsonl(content)
        assert len(events) == 1
        assert events[0]["type"] == "Message"

    def test_empty(self):
        assert extract_json_events_jsonl("") == []

    def test_from_fixture(self, sample_jsonl_path):
        content = sample_jsonl_path.read_text(encoding="utf-8")
        events = extract_json_events_jsonl(content)
        assert len(events) == 11
        types = [e["type"] for e in events]
        assert "MessageEvent" in types
        assert "LLMStreamStartEvent" in types
        assert "AgentConstructionExceptionEvent" in types


# ------------------------------------------------------------------
# extract_json_events_json_array
# ------------------------------------------------------------------


class TestExtractJsonEventsJsonArray:
    def test_basic(self):
        content = json.dumps([
            {"type": "Message", "sender": "A"},
            {"type": "ToolCall", "tool": "t"},
        ])
        events = extract_json_events_json_array(content)
        assert len(events) == 2
        assert events[0]["type"] == "Message"

    def test_filters_objects_without_type(self):
        content = json.dumps([
            {"data": "no type"},
            {"type": "Message"},
        ])
        events = extract_json_events_json_array(content)
        assert len(events) == 1

    def test_malformed_json(self):
        assert extract_json_events_json_array("[bad json") == []

    def test_not_a_list(self):
        assert extract_json_events_json_array('{"type": "Message"}') == []

    def test_empty_array(self):
        assert extract_json_events_json_array("[]") == []

    def test_from_fixture(self, sample_json_array_path):
        content = sample_json_array_path.read_text(encoding="utf-8")
        events = extract_json_events_json_array(content)
        assert len(events) == 11
        types = [e["type"] for e in events]
        assert "MessageDroppedEvent" in types
        assert "LLMStreamEndEvent" in types


# ------------------------------------------------------------------
# parse_events (unified entry point)
# ------------------------------------------------------------------


class TestParseEvents:
    def test_dispatches_to_jsonl(self):
        content = '{"type": "Message", "sender": "A"}\n{"type": "ToolCall", "tool": "t"}\n'
        events = parse_events(content)
        assert len(events) == 2

    def test_dispatches_to_json_array(self):
        content = json.dumps([{"type": "Message"}, {"type": "ToolCall", "tool": "t"}])
        events = parse_events(content)
        assert len(events) == 2

    def test_dispatches_to_text(self):
        content = (
            "2026-01-15 log line\n"
            '{\n  "type": "Message",\n  "sender": "A"\n}\n'
        )
        events = parse_events(content)
        assert len(events) == 1
        assert events[0]["type"] == "Message"

    def test_empty(self):
        assert parse_events("") == []

    def test_backward_compat_with_fixture(self, sample_log_content):
        """parse_events should produce the same results as extract_json_events
        when given text-format content."""
        from agent_eval.adapters.autogen.event_parser import extract_json_events

        expected = extract_json_events(sample_log_content)
        actual = parse_events(sample_log_content)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a["type"] == e["type"]
