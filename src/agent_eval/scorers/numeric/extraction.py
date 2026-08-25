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
# Percentage range patterns like "5.5% to 7.5%" or "16.1%-19.1%"
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
_DATETIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)?"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
)


def _mask_non_quantities(text: str) -> str:
    """Blank out URL, date and citation regions, preserving every offset.

    Replaces with spaces rather than deleting: the callers classify each number
    by the text *around* its match position, so shifting offsets would silently
    misread the approximate/derived context of every later number.
    """

    def _blank(match: re.Match) -> str:
        return " " * (match.end() - match.start())

    masked = _URL_RE.sub(_blank, text)
    masked = _DATETIME_RE.sub(_blank, masked)
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
        window = stripped[max(0, pos - 60) : pos]
        return bool(_DERIVED_KEYWORDS_RE.search(window))

    def _classify(pos: int) -> dict:
        approx = _has_approx_prefix(pos)
        derived = _has_derived_context(pos)
        in_range = _in_range(pos)
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
        info = _classify(match.start())
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
            info = _classify(match.start())
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
