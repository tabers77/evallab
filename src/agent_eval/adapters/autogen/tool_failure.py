"""Tool failure detection for AutoGen ToolCall results.

Ported from log_evaluator.py:_is_tool_call_failed — two-tier detection:
  Tier 1: Structured/error-shaped payloads (dict keys)
  Tier 2: String-based error indicators (case-insensitive)
"""

from __future__ import annotations

import re
from typing import Any

_ERROR_KEYS = frozenset({"error", "errors", "exception", "traceback"})

_STRING_ERROR_PATTERNS = [
    "validation error",
    "toolexecutionerror",
    "tool execution error",
    "traceback",
    "exception",
    "http 4",
    "http 5",
]

_FALSE_POSITIVE_RE = re.compile(r"\b(no|without)\s+error\b")
_ERROR_WORD_RE = re.compile(r"\berror\b")


def is_tool_call_failed(tool_result: Any) -> bool:
    """Determine whether a ToolCall result represents a failure.

    Tier 1 — structured error payload (dict keys):
      - Keys: error, errors, exception, traceback
      - success=False, ok=False, status="error"
      - status_code >= 400

    Tier 2 — string-based fallback:
      - Known error patterns (case-insensitive)
      - Word "error" with false-positive guard ("no error")
    """
    # ---------- Tier 1: structured error payload ----------
    if isinstance(tool_result, dict):
        if _ERROR_KEYS & tool_result.keys():
            return True
        if tool_result.get("success") is False:
            return True
        if tool_result.get("ok") is False:
            return True
        if tool_result.get("status") == "error":
            return True
        if tool_result.get("status_code", 0) >= 400:
            return True

    # ---------- Tier 2: string-based fallback ----------
    if isinstance(tool_result, str):
        text = tool_result.lower()
        if any(p in text for p in _STRING_ERROR_PATTERNS):
            return True
        if _ERROR_WORD_RE.search(text) and not _FALSE_POSITIVE_RE.search(text):
            return True

    return False
