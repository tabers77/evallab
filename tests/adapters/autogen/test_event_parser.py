"""Tests for adapters.autogen.event_parser — JSON event extraction."""

from agent_eval.adapters.autogen.event_parser import extract_json_events


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
