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
