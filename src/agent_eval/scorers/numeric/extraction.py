"""Number extraction utilities for hallucination detection.

Ported from numeric_validator.py — extracts numeric values from text
and from ToolCall event results.
"""

from __future__ import annotations

import re
from typing import Any


_CURRENCY_RE = re.compile(r"[\u20ac$\u00a3\u00a5]")  # euro, dollar, pound, yen

# Million/Billion/Thousand notation: 283M, 283 million, 5.5B, 500K
_MBK_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*([MmBbKk])(?:illion)?\b")
_MBK_MULTIPLIERS = {"m": 1e6, "b": 1e9, "k": 1e3}

# Standard numbers: comma/dot-separated (283,399,382.94) OR plain (283399382.94)
_NUMBER_RE = re.compile(
    r"\b\d{1,3}(?:[,.]\d{3})*(?:[.,]\d+)?\b"  # comma/dot-grouped
    r"|\b\d+\.\d+\b"  # plain decimal (e.g. 283399382.94)
    r"|\b\d{4,}\b"  # plain large integer (e.g. 283399382)
)

# Patterns indicating a number is derived/calculated, not a direct tool citation
_APPROX_PREFIX_RE = re.compile(r"~\s*$")
_DERIVED_KEYWORDS_RE = re.compile(
    r"\b(?:approximately|approx|about|around|roughly|estimated?|"
    r"save|reduce(?:\s+by)?|target|scenario|projected?|"
    r"assume|potential|up\s+to|at\s+least)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Arithmetic-result context.
#
# The approximation/projection vocabulary above had nothing for a number the
# agent *computed* from figures it correctly quoted. Observed on a production
# orchestrate run (intelligence-platform, 2026-08-26): the answer quoted 2024
# revenue 113,851,699.92 and 2025 revenue 118,294,376.54 from the tool output,
# then wrote the delta as "+4.44m". That delta was reported as CRITICAL
# "Data Fabrication ... closest 12,845,189.28, error 65.4%" and pulled a
# correct answer to grade D, while the platform's own fact-checker passed it as
# a "reasonable inference". A delta is the most ordinary thing an analytical
# answer computes, so this was never an edge case.
#
# Split into two patterns rather than one keyword list, because the two carry
# different amounts of evidence:
#
#   1. Result NOUNS are inherently the output of an operation. "the difference
#      is X", "YoY change ... X" — X cannot be a verbatim tool value.
#
#   2. Change VERBS are only evidence when paired with "by"/"of". This is the
#      distinction a single keyword list cannot express, and getting it wrong
#      is what makes the scorer permissive: "revenue increased BY 4,442,676.62"
#      is a delta, but "revenue increased TO 118,294,376.54" quotes a real
#      figure and must still be checked. A bare "increase" alternative excuses
#      both.
#
# Metric names and bare comparatives are deliberately absent. An early draft
# included "margin", "net", "higher" and "lower"; "Our EBITDA margin was
# 47,000,000.00" then stopped being flagged. A keyword that also names a metric
# silences the scorer on exactly the sentences it exists to check.
_DELTA_NOUNS_RE = re.compile(
    r"\b(?:change[ds]?|delta|difference|variance|uplift|"
    r"yoy|y/y|year[- ]over[- ]year|year[- ]on[- ]year)\b",
    re.IGNORECASE,
)
# Verb + "by"/"of", anchored to the end of the window so it must sit
# immediately before the number.
_DELTA_VERB_RE = re.compile(
    r"\b(?:increas|decreas|grow|grew|grown|ros[e]|rise|rising|f[ae]ll|"
    r"declin|drop|gain|improv|expand|shrank|shrunk|shrink)\w*\s+"
    r"(?:by|of)\s*$",
    re.IGNORECASE,
)

# An explicit leading "+" marks a computed change, never a quoted figure.
# Tool results are raw values — they never carry a plus sign — so "+4.44m" is
# by construction something the agent worked out rather than read. Only "+" is
# treated this way: a leading "-" is ambiguous, because a tool legitimately
# returns negative margins and losses that an answer would quote verbatim.
_SIGNED_DELTA_RE = re.compile(r"\+\s*$")
_PERCENT_RANGE_RE = re.compile(
    r"\d+(?:\.\d+)?%?\s*(?:to|-)\s*\d+(?:\.\d+)?%"
)

# Regions whose digits are not quantities and must never be read as figures the
# agent claimed. Both were observed producing CRITICAL "numeric_fabrication"
# issues on real document-grounded answers, one of which scored 0.0:
#   - URLs and percent-escapes: "...Packaging%20Solutions%20%26%20Design.txt"
#     yielded 20 and 26, reported as fabricated monetary values.
#   - Citation markers: "...Emails/**.[1][2][3]" yielded 1, 2 and 3.
# Any answer citing sources or linking a file hits this, so it is not an edge
# case — it is the normal shape of a retrieval-grounded answer.
_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\S*%[0-9A-Fa-f]{2}\S*")
_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")
# Dates and clock times. "2026-08-25T09:13:06Z" was reported as a fabricated
# 13, and "2026-04-28" as a fabricated 28 — a date is a label, never a
# quantity the agent claimed to have read from a tool.
# Product names carrying digits. "extend the search to m365" was reported as
# a fabricated 365.00, and "Q2" as a fabricated 2 — a product or label is not
# a quantity. Matches a digit run glued to letters (M365, GPT4, Q2) and the
# common spaced Microsoft forms.
_PRODUCT_RE = re.compile(
    r"\b(?:microsoft|office|teams|sharepoint|windows|dynamics|azure)\s+\d{3,4}\b"
    r"|\b[A-Za-z]+\d+\b"
    , re.IGNORECASE,
)

_DATETIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)?"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
)

# Ordered-list markers. The "1." and "2." that open the steps of a numbered
# list are labels, exactly like a citation marker, and they are the last shape
# of prose punctuation still read as data after 0.3.2 masked citations.
#
# Observed on intelligence-platform run f5d923ada3cf (2026-09-01) against this
# build, 0.3.6. A solve answer whose bound data source returned zero rows
# correctly reported that it could not compute the figure, and closed with:
#
#     1. identify which `period_year` values exist
#     2. rerun the contribution margin query for those years
#
# Those markers were the only 1 and 2 in the answer. Both were reported as
# CRITICAL "Data Fabrication" -- `Number 1.00 not found in tool results
# (closest: 609.00, error: 99.8%)` -- dropping numeric_accuracy to 0.6 and the
# grade to 52.0, on an answer that invented nothing and said so. The platform's
# own fact-checker returned PASS on the same text and the LLM judge scored
# groundedness 0.98; this scorer was the only dissenting voice, and what it was
# reading was list punctuation.
#
# `min_value=1.0` does not catch these: the guard is `abs(n) < min_value`, so
# 1.0 and 2.0 both pass it. Raising min_value would silence real small
# quantities instead, which is why this is a mask and not a threshold change.
#
# Deliberately narrow, so table rows and decimals survive:
#   - the digits must OPEN the line (after optional quote/bullet indent), and
#   - be followed by "." or ")" AND then whitespace.
# "0.85 is the ratio" is untouched (no space after the dot) and the table row
# "2024 | 113,851,699.92" is untouched (no dot or paren after 2024).
_LIST_MARKER_RE = re.compile(
    r"^[ \t>*+\-]*\(?\d{1,3}[.)](?=\s)",
    re.MULTILINE,
)


def _mask_non_quantities(text: str) -> str:
    """Blank out URL, date, citation and list-marker regions, preserving offsets.

    Replaces with spaces rather than deleting: the callers classify each number
    by the text *around* its match position, so shifting offsets would silently
    misread the approximate/derived context of every later number.
    """

    def _blank(match: re.Match) -> str:
        return " " * (match.end() - match.start())

    masked = _URL_RE.sub(_blank, text)
    masked = _PRODUCT_RE.sub(_blank, masked)
    masked = _DATETIME_RE.sub(_blank, masked)
    masked = _LIST_MARKER_RE.sub(_blank, masked)
    return _CITATION_RE.sub(_blank, masked)


def extract_numbers_from_text(text: str) -> list[float]:
    """Extract all numeric values from text.

    Handles:
      - Standard decimals: ``283,399,382.94``
      - European format: ``283.399.382,94``
      - Million notation: ``283M``, ``283 million``
      - Billion notation: ``5.5B``, ``5.5 billion``
      - Thousand notation: ``500K``
      - With currencies: ``\\u20ac283M``, ``$5.5B``
    """
    numbers: list[float] = []

    text = _CURRENCY_RE.sub("", text)

    # Pattern 1: M/B/K notation (process first)
    mbk_positions: list[tuple[int, int]] = []
    for match in _MBK_RE.finditer(text):
        num_str = match.group(1).replace(",", "")
        value = float(num_str)
        unit = match.group(2).lower()
        numbers.append(value * _MBK_MULTIPLIERS[unit])
        mbk_positions.append((match.start(), match.end()))

    # Pattern 2: Standard numbers (skip those already covered by M/B/K)
    for match in _NUMBER_RE.finditer(text):
        in_mbk = any(start <= match.start() < end for start, end in mbk_positions)
        if in_mbk:
            continue

        num_str = match.group().replace(",", "")
        try:
            numbers.append(float(num_str))
        except ValueError:
            pass

    return numbers


def extract_numbers_with_context(text: str) -> list[dict]:
    """Extract numbers from text with surrounding context for classification.

    Returns a list of dicts with keys:
      - ``value``: the numeric value
      - ``is_approximate``: True if preceded by ``~`` or approximate keywords
      - ``is_derived``: True if context suggests a calculation or proposal
        (approximation/projection wording, arithmetic-result wording such as
        "YoY change" or "increase", or an explicit "+" sign on the number)
      - ``in_percent_range``: True if part of a percentage range pattern
    """
    results: list[dict] = []
    stripped = _mask_non_quantities(_CURRENCY_RE.sub("", text))

    # Build a set of positions covered by percent-range patterns
    range_spans: list[tuple[int, int]] = []
    for m in _PERCENT_RANGE_RE.finditer(stripped):
        range_spans.append((m.start(), m.end()))

    def _in_range(pos: int) -> bool:
        return any(s <= pos < e for s, e in range_spans)

    def _has_approx_prefix(pos: int) -> bool:
        preceding = stripped[max(0, pos - 5) : pos]
        return bool(_APPROX_PREFIX_RE.search(preceding))

    def _has_derived_context(pos: int) -> bool:
        # Clamp the lookback to the number's own line. A raw 60-character
        # window silently crosses line and table-row boundaries, so a keyword
        # belonging to one bullet leaks onto the next one's figures: in the
        # answer that motivated the delta keywords above, "about" on the
        # "+4.59m, about +19.7%" bullet sat within 60 characters of the
        # 113,851,699.92 in the table row below it, marking a verbatim tool
        # value as derived. Keywords qualify the number they share a line with;
        # anything further away is coincidence, and treating it as intent makes
        # the scorer permissive exactly where it should bite.
        line_start = stripped.rfind("\n", 0, pos) + 1
        window = stripped[max(line_start, pos - 60) : pos]
        return bool(
            _DERIVED_KEYWORDS_RE.search(window)
            or _DELTA_NOUNS_RE.search(window)
            or _DELTA_VERB_RE.search(window)
        )

    def _has_signed_prefix(pos: int) -> bool:
        """A "+" immediately before the number means the agent computed it."""
        return bool(_SIGNED_DELTA_RE.search(stripped[max(0, pos - 3) : pos]))

    def _is_percent(end: int) -> bool:
        """A lone "3.6%" is a rate, not a figure quotable from a tool result.

        Only percent *ranges* were excluded before, so a single percentage was
        compared against raw tool numbers and reported as fabricated — observed
        on "3.6%" (closest match 4.00) and "25% to 30%".
        """
        return stripped[end : end + 2].lstrip().startswith("%")

    def _classify(pos: int, end: int | None = None) -> dict:
        approx = _has_approx_prefix(pos)
        derived = _has_derived_context(pos) or _has_signed_prefix(pos)
        in_range = _in_range(pos) or (end is not None and _is_percent(end))
        return {
            "is_approximate": approx,
            "is_derived": derived or approx,
            "in_percent_range": in_range,
        }

    # M/B/K notation
    mbk_positions: list[tuple[int, int]] = []
    for match in _MBK_RE.finditer(stripped):
        num_str = match.group(1).replace(",", "")
        value = float(num_str)
        unit = match.group(2).lower()
        info = _classify(match.start(), match.end())
        info["value"] = value * _MBK_MULTIPLIERS[unit]
        results.append(info)
        mbk_positions.append((match.start(), match.end()))

    # Standard numbers
    for match in _NUMBER_RE.finditer(stripped):
        in_mbk = any(start <= match.start() < end for start, end in mbk_positions)
        if in_mbk:
            continue
        num_str = match.group().replace(",", "")
        try:
            info = _classify(match.start(), match.end())
            info["value"] = float(num_str)
            results.append(info)
        except ValueError:
            pass

    return results


def extract_numbers_from_tool_results(
    events: list[dict],
) -> dict[str, list[float]]:
    """Extract numbers from ToolCall event results.

    Parameters
    ----------
    events
        List of parsed JSON events (from :func:`extract_json_events`).

    Returns
    -------
    dict
        Mapping from tool name to list of numbers found in that tool's results.
    """
    tool_numbers: dict[str, list[float]] = {}

    tool_calls = [e for e in events if e.get("type") == "ToolCall"]

    for tc in tool_calls:
        tool_name = tc.get("tool") or tc.get("tool_name", "unknown")
        result = tc.get("result", "")

        if tool_name not in tool_numbers:
            tool_numbers[tool_name] = []

        if isinstance(result, dict):
            tool_numbers[tool_name].extend(_extract_from_dict(result))
        else:
            tool_numbers[tool_name].extend(extract_numbers_from_text(str(result)))

    return tool_numbers


def extract_answer_block(content: str) -> str:
    """Extract the ``<ANSWER>:`` block from log content."""
    answer_start = content.find("<ANSWER>:")
    if answer_start < 0:
        return ""

    answer_end = content.find("</ANSWER>", answer_start)
    if answer_end < 0:
        answer_end = content.find("\u2500" * 16, answer_start)
        if answer_end < 0:
            answer_end = len(content)

    return content[answer_start:answer_end]


def _extract_from_dict(obj: Any) -> list[float]:
    """Recursively extract numeric values from nested dicts/lists."""
    numbers: list[float] = []
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, (int, float)):
                numbers.append(float(value))
            elif isinstance(value, dict):
                numbers.extend(_extract_from_dict(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (int, float)):
                        numbers.append(float(item))
                    elif isinstance(item, dict):
                        numbers.extend(_extract_from_dict(item))
    return numbers
