"""JSON event extraction from AutoGen log files.

Ported from log_evaluator.py:_extract_json_events — uses a line-by-line
parser with brace-counting state machine to handle multi-line JSON.
"""

from __future__ import annotations

import json


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
