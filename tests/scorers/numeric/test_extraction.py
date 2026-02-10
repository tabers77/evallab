"""Tests for scorers.numeric.extraction — number extraction utilities."""

import pytest

from agent_eval.scorers.numeric.extraction import (
    extract_answer_block,
    extract_numbers_from_text,
    extract_numbers_from_tool_results,
)


class TestExtractNumbersFromText:
    def test_simple_integer(self):
        nums = extract_numbers_from_text("The value is 42")
        assert 42.0 in nums

    def test_decimal(self):
        nums = extract_numbers_from_text("Revenue was 283399382.94")
        assert 283399382.94 in nums

    def test_comma_separated(self):
        nums = extract_numbers_from_text("Revenue: 283,399,382.94")
        assert pytest.approx(283399382.94) in nums

    def test_million_notation(self):
        nums = extract_numbers_from_text("Revenue is 283M")
        assert 283e6 in nums

    def test_billion_notation(self):
        nums = extract_numbers_from_text("Market cap: 5.5B")
        assert 5.5e9 in nums

    def test_thousand_notation(self):
        nums = extract_numbers_from_text("Volume: 500K units")
        assert 500e3 in nums

    def test_currency_symbols_stripped(self):
        nums = extract_numbers_from_text("Price: $150 or \u20ac200")
        assert 150.0 in nums
        assert 200.0 in nums

    def test_million_word_notation(self):
        nums = extract_numbers_from_text("Revenue was 283 million")
        assert 283e6 in nums

    def test_no_double_extraction_for_mbk(self):
        """Numbers in M/B/K notation should not be extracted twice."""
        nums = extract_numbers_from_text("Revenue is 283M")
        # Should have 283000000, not also 283
        assert 283e6 in nums
        assert 283.0 not in nums

    def test_empty_text(self):
        assert extract_numbers_from_text("") == []

    def test_no_numbers(self):
        assert extract_numbers_from_text("No numbers here") == []


class TestExtractNumbersFromToolResults:
    def test_dict_result(self):
        events = [
            {
                "type": "ToolCall",
                "tool": "get_finances",
                "result": {"REVENUE": 283399382.94, "VOLUME": 15000},
            }
        ]
        nums = extract_numbers_from_tool_results(events)
        assert "get_finances" in nums
        assert 283399382.94 in nums["get_finances"]
        assert 15000.0 in nums["get_finances"]

    def test_string_result(self):
        events = [
            {
                "type": "ToolCall",
                "tool": "ask_web",
                "result": "The company has 500 employees and $2.5B revenue",
            }
        ]
        nums = extract_numbers_from_tool_results(events)
        assert "ask_web" in nums
        assert 500.0 in nums["ask_web"]
        assert 2.5e9 in nums["ask_web"]

    def test_nested_dict_result(self):
        events = [
            {
                "type": "ToolCall",
                "tool": "get_data",
                "result": {
                    "financials": {
                        "revenue": 100000.0,
                        "costs": 80000.0,
                    }
                },
            }
        ]
        nums = extract_numbers_from_tool_results(events)
        assert 100000.0 in nums["get_data"]
        assert 80000.0 in nums["get_data"]

    def test_filters_non_tool_events(self):
        events = [
            {"type": "Message", "content": "Numbers: 42"},
            {"type": "ToolCall", "tool": "t1", "result": {"val": 99}},
        ]
        nums = extract_numbers_from_tool_results(events)
        assert "t1" in nums
        assert len(nums) == 1  # Only tool events

    def test_empty_events(self):
        assert extract_numbers_from_tool_results([]) == {}


class TestExtractAnswerBlock:
    def test_with_answer_tags(self):
        content = "prefix\n<ANSWER>: The answer is 42\n</ANSWER>\nsuffix"
        block = extract_answer_block(content)
        assert "<ANSWER>:" in block
        assert "42" in block

    def test_without_answer_tags(self):
        content = "No answer here"
        assert extract_answer_block(content) == ""

    def test_unclosed_answer_tag(self):
        content = "<ANSWER>: Some answer without closing tag\nMore text"
        block = extract_answer_block(content)
        assert "Some answer" in block
