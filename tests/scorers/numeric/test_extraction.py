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


class TestProductNamesAndPercentages:
    """More shapes that are not figures quoted from a tool result.

    From production 0.3.3 runs: "extend the search to m365" was reported as a
    fabricated 365.00, and "3.6%" as a fabricated 3.60 with closest match 4.00.
    """

    def test_glued_product_name(self):
        assert extract_numbers_with_context("extend the search to m365") == []

    def test_spaced_microsoft_product(self):
        assert extract_numbers_with_context("the Microsoft 365 folder") == []

    def test_quarter_label(self):
        values = [e["value"] for e in extract_numbers_with_context("Q2 revenue rose")]
        assert values == []

    def test_single_percentage_is_flagged_as_percent(self):
        entries = extract_numbers_with_context("a 3.6% rise year on year")
        assert len(entries) == 1
        assert entries[0]["value"] == 3.6
        # Consumers skip percent entries, so this is what stops the false CRITICAL.
        assert entries[0]["in_percent_range"] is True

    def test_percent_ranges_still_flagged(self):
        entries = extract_numbers_with_context("25% to 30% range")
        assert all(e["in_percent_range"] for e in entries)

    def test_a_plain_figure_is_not_a_percent(self):
        entries = extract_numbers_with_context("Kraftliner at 795.09 per tonne")
        assert len(entries) == 1
        assert entries[0]["in_percent_range"] is False


class TestDerivedDeltaNotFabrication:
    """A number the agent computed from figures it correctly quoted is not a
    fabrication.

    From a production orchestrate run (intelligence-platform, 2026-08-26). The
    answer quoted 2024 revenue 113,851,699.92 and 2025 revenue 118,294,376.54
    from `get_product_group_finances`, then wrote the delta as "+4.44m". That
    delta was reported as CRITICAL "Data Fabrication ... closest
    12,845,189.28, error 65.4%" and pulled a correct answer down to grade D,
    while the platform's own fact-checker passed the same figure as a
    "reasonable inference". Every YoY/variance answer hits this path.
    """

    ANSWER = (
        "### YoY change, 2025 vs 2024\n"
        "- **Revenue:** +4.44m, about **+3.9%**\n"
        "- **EBITDA:** +4.59m, about **+19.7%**\n"
        "\n"
        "| 2024 | Liquid | 113,851,699.92 | 23,329,563.57 |\n"
        "| 2025 | Liquid | 118,294,376.54 | 27,922,517.71 |\n"
    )

    def _entry(self, text, value):
        for e in extract_numbers_with_context(text):
            if abs(e["value"] - value) < 0.01:
                return e
        raise AssertionError(f"{value} not extracted from {text!r}")

    def test_signed_delta_is_derived(self):
        assert self._entry(self.ANSWER, 4_440_000.0)["is_derived"] is True
        assert self._entry(self.ANSWER, 4_590_000.0)["is_derived"] is True

    def test_quoted_tool_values_are_still_checked(self):
        """The counterpart guard. A keyword must not leak onto neighbouring
        figures: "about" on the EBITDA bullet sits within 60 characters of
        113,851,699.92 in the table row below, so before the lookback was
        clamped to the number's own line these verbatim tool values were
        excused too — which would silence the scorer on the very numbers it
        exists to verify."""
        for value in (113_851_699.92, 23_329_563.57, 118_294_376.54, 27_922_517.71):
            assert self._entry(self.ANSWER, value)["is_derived"] is False

    def test_arithmetic_wording_without_a_sign(self):
        for text in (
            "Revenue increased by 4,442,676.62.",
            "The difference is 4,442,676.62.",
            "Revenue grew by 4,442,676.62 versus last year.",
        ):
            assert self._entry(text, 4_442_676.62)["is_derived"] is True

    def test_change_verb_with_to_still_quotes_a_real_figure(self):
        """The distinction a flat keyword list cannot express, and the reason
        the verbs are matched only when followed by "by"/"of". "increased BY X"
        is a delta; "increased TO X" quotes a tool value and must stay checked.
        A bare "increase" alternative excuses both, which would silence the
        scorer on most trend sentences."""
        for text, value in (
            ("Revenue increased to 118,294,376.54 in 2025.", 118_294_376.54),
            ("Revenue rose to 118,294,376.54.", 118_294_376.54),
        ):
            assert self._entry(text, value)["is_derived"] is False

    def test_metric_names_do_not_excuse_a_fabrication(self):
        """Regression on the fix itself. "margin", "net", "higher" and "lower"
        were in the first draft of the derived-keyword list; because they name
        metrics rather than operations, they stopped genuinely invented figures
        from being flagged."""
        for text, value in (
            ("Our EBITDA margin was 47,000,000.00 last year.", 47_000_000.0),
            ("Net sales reached 88,000,000.00.", 88_000_000.0),
            ("Spend was higher at 231,198,548.11.", 231_198_548.11),
            ("Total spend was 231,198,548.11.", 231_198_548.11),
        ):
            assert self._entry(text, value)["is_derived"] is False

    def test_a_leading_minus_is_not_treated_as_a_delta(self):
        """Only "+" implies a computation. A tool legitimately returns negative
        margins and losses that an answer quotes verbatim, so "-" must keep
        being checked."""
        # The extractor reports magnitude; the sign lives in the surrounding
        # text, which is exactly what _has_signed_prefix inspects.
        entry = self._entry("Result was -4,442,676.62 for the year.", 4_442_676.62)
        assert entry["is_derived"] is False

