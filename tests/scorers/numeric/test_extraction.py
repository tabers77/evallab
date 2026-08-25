"""Tests for scorers.numeric.extraction — number extraction utilities."""

import pytest

from agent_eval.scorers.numeric.extraction import (
    extract_answer_block,
    extract_numbers_from_text,
    extract_numbers_from_tool_results,
    extract_numbers_with_context,
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


class TestNonQuantityRegionsIgnored:
    """Digits in URLs and citation markers are not figures the agent claimed.

    Both shapes below were observed producing CRITICAL "numeric_fabrication"
    issues on real document-grounded answers in production — one scored 0.0
    (grade F) purely on citation markers and URL percent-escapes, on a question
    that asked whether a SharePoint file was visible and contained no monetary
    figures at all.
    """

    def test_percent_escapes_in_a_filename_are_not_numbers(self):
        text = "See Valona%20Insights_%20Packaging%20Solutions%20%26%20Design.txt"
        assert extract_numbers_with_context(text) == []

    def test_citation_markers_are_not_numbers(self):
        text = "The folder is Documents/CopilotAgent/Emails/.[1][2][3]"
        assert extract_numbers_with_context(text) == []

    def test_multi_number_citation_group(self):
        assert extract_numbers_with_context("Confirmed by sources [1, 2; 3].") == []

    def test_http_url_digits_are_not_numbers(self):
        text = "Source: https://example.com/reports/2024/q3?id=98765"
        assert extract_numbers_with_context(text) == []

    def test_real_figures_beside_a_url_still_extract(self):
        """Masking must not swallow the quantities the scorer exists to check."""
        text = "Revenue was 254649476.24 per https://example.com/a%20b.txt"
        values = [e["value"] for e in extract_numbers_with_context(text)]
        assert 254649476.24 in values

    def test_real_figures_beside_a_citation_still_extract(self):
        text = "Contribution margin was 94384457.24 last year.[1]"
        values = [e["value"] for e in extract_numbers_with_context(text)]
        assert 94384457.24 in values

    def test_masking_preserves_derived_context(self):
        """Offsets must not shift, or the approximate/derived flags misread."""
        text = "See https://example.com/a%20b — approximately 1234.5 tonnes"
        entries = extract_numbers_with_context(text)
        assert len(entries) == 1
        assert entries[0]["value"] == 1234.5
        assert entries[0]["is_derived"] is True


class TestDatesAndTimesIgnored:
    """A date is a label, not a quantity read from a tool.

    "2026-08-25T09:13:06Z" was reported in production as a fabricated 13 and
    "2026-04-28" as a fabricated 28, both CRITICAL.
    """

    def test_iso_timestamp(self):
        text = "Last modified: 2026-08-25T09:13:06Z"
        assert extract_numbers_with_context(text) == []

    def test_plain_iso_date(self):
        assert extract_numbers_with_context("PIX newsletter, 2026-04-28.") == []

    def test_clock_time(self):
        assert extract_numbers_with_context("Delivered at 09:13.") == []

    def test_slash_date(self):
        assert extract_numbers_with_context("Dated 28/04/2026 in the file.") == []

    def test_a_price_beside_a_date_still_extracts(self):
        text = "On 2026-04-28 Kraftliner was 795.09 per tonne"
        values = [e["value"] for e in extract_numbers_with_context(text)]
        assert values == [795.09]


class TestPatternsAreNotSilentlyCorrupted:
    """A regex holding a control character matches nothing and fails silently.

    `\b` written through a non-raw string becomes a literal backspace (0x08),
    which compiles fine, greps invisibly, and disables the pattern completely —
    exactly how the date mask above shipped inert on its first attempt.
    """

    def test_no_control_characters_in_module_patterns(self):
        import re as _re

        from agent_eval.scorers.numeric import extraction as mod

        for name in dir(mod):
            if not name.endswith("_RE"):
                continue
            obj = getattr(mod, name)
            if not isinstance(obj, _re.Pattern):
                continue
            offenders = [
                hex(ord(ch)) for ch in obj.pattern if ord(ch) < 32 and ch != "\n"
            ]
            assert not offenders, f"{name} contains control chars {offenders}"
