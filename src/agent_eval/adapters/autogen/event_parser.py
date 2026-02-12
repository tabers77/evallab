"""JSON event extraction from AutoGen log files.

Supports three input formats:

- **text** — AutoGen event.txt with interleaved log lines and multi-line
  JSON blocks (brace-counting state machine).
- **jsonl** — One JSON object per line (structured event logger output).
- **json_array** — A single JSON array of event objects.

Use :func:`parse_events` as the unified entry point; it auto-detects the
format and dispatches to the appropriate parser.
"""

from __future__ import annotations

import json


# ------------------------------------------------------------------
# Format detection
# ------------------------------------------------------------------


def detect_format(content: str) -> str:
    """Detect the format of AutoGen log content.

    Returns ``"json_array"``, ``"jsonl"``, or ``"text"``.

    Detection logic:
    - Starts with ``[`` (after stripping) → ``json_array``
    - First non-empty line is a complete JSON object → ``jsonl``
    - Otherwise → ``text`` (existing brace-counting parser)
    """
    stripped = content.lstrip()
    if stripped.startswith("["):
        return "json_array"

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                json.loads(line)
                return "jsonl"
            except (json.JSONDecodeError, ValueError):
                pass
        break  # only inspect the first non-empty line

    return "text"


# ------------------------------------------------------------------
# JSONL parser
# ------------------------------------------------------------------


def extract_json_events_jsonl(content: str) -> list[dict]:
    """Extract JSON events from one-JSON-per-line (JSONL) content.

    Each non-empty line is parsed independently. Only dicts with a
    ``"type"`` key are returned; malformed lines are silently skipped.
    """
    events: list[dict] = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "type" in obj:
                events.append(obj)
        except (json.JSONDecodeError, ValueError):
            pass
    return events


# ------------------------------------------------------------------
# JSON Array parser
# ------------------------------------------------------------------


def extract_json_events_json_array(content: str) -> list[dict]:
    """Extract JSON events from a JSON array (``[{...}, ...]``).

    Only dicts with a ``"type"`` key are returned.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(data, list):
        return []

    return [obj for obj in data if isinstance(obj, dict) and "type" in obj]


# ------------------------------------------------------------------
# Unified entry point
# ------------------------------------------------------------------


def parse_events(content: str) -> list[dict]:
    """Auto-detect format and extract JSON events.

    Delegates to :func:`extract_json_events` (text),
    :func:`extract_json_events_jsonl`, or
    :func:`extract_json_events_json_array` based on
    :func:`detect_format`.
    """
    fmt = detect_format(content)
    if fmt == "json_array":
        return extract_json_events_json_array(content)
    if fmt == "jsonl":
        return extract_json_events_jsonl(content)
    return extract_json_events(content)


# ------------------------------------------------------------------
# Text parser (original brace-counting state machine)
# ------------------------------------------------------------------


def extract_json_events(content: str) -> list[dict]:
    """Extract JSON event objects from AutoGen log content.

    Uses a brace-counting state machine that handles multi-line JSON
    blocks such as::

        2026-01-15 14:04:11,206 autogen_core.events
        {
          "type": "Message",
          "sender": "FinanceExpert",
          ...
        }

    Only dicts with a ``"type"`` key are returned; malformed JSON is
    silently skipped.
    """
    events: list[dict] = []
    lines = content.split("\n")

    in_json = False
    brace_count = 0
    current_json: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not in_json and stripped.startswith("{"):
            in_json = True
            brace_count = 0
            current_json = []

        if in_json:
            current_json.append(line)
            brace_count += stripped.count("{") - stripped.count("}")

            if brace_count == 0:
                try:
                    json_str = "\n".join(current_json)
                    event = json.loads(json_str)
                    if isinstance(event, dict) and "type" in event:
                        events.append(event)
                except (json.JSONDecodeError, ValueError):
                    pass
                finally:
                    in_json = False
                    current_json = []

    return events
