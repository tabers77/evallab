"""OTel span → agent-eval Step mapping logic.

Handles OTLP JSON attribute unwrapping and maps ``gen_ai.*`` semantic
convention attributes to canonical :class:`Step` fields.
"""

from __future__ import annotations

from typing import Any

from agent_eval.core.models import Step, StepKind


# ── gen_ai.operation.name → StepKind ────────────────────────────────

_OPERATION_TO_KIND: dict[str, StepKind] = {
    "chat": StepKind.LLM_CALL,
    "text_completion": StepKind.LLM_CALL,
    "generate_content": StepKind.LLM_CALL,
    "execute_tool": StepKind.TOOL_CALL,
    "invoke_agent": StepKind.MESSAGE,
    "create_agent": StepKind.MESSAGE,
    "embeddings": StepKind.CUSTOM,
    "retrieval": StepKind.CUSTOM,
}

# Span name substrings used as fallback when operation.name is absent.
_NAME_HINTS: list[tuple[str, StepKind]] = [
    ("llm", StepKind.LLM_CALL),
    ("chat", StepKind.LLM_CALL),
    ("completion", StepKind.LLM_CALL),
    ("tool", StepKind.TOOL_CALL),
    ("agent", StepKind.MESSAGE),
]


# ── OTLP JSON value unwrapping ──────────────────────────────────────

def unwrap_attribute_value(attr_value: dict | Any) -> Any:
    """Unwrap an OTLP JSON attribute value wrapper.

    OTLP JSON encodes attribute values as, e.g.::

        {"stringValue": "gpt-4o"}
        {"intValue": "42"}            # note: encoded as string
        {"doubleValue": 3.14}
        {"boolValue": true}
        {"arrayValue": {"values": [...]}}
        {"kvlistValue": {"values": [{"key": ..., "value": ...}, ...]}}

    Returns the unwrapped Python value.
    """
    if not isinstance(attr_value, dict):
        return attr_value

    if "stringValue" in attr_value:
        return attr_value["stringValue"]
    if "intValue" in attr_value:
        raw = attr_value["intValue"]
        return int(raw) if isinstance(raw, str) else raw
    if "doubleValue" in attr_value:
        return attr_value["doubleValue"]
    if "boolValue" in attr_value:
        return attr_value["boolValue"]
    if "arrayValue" in attr_value:
        values = attr_value["arrayValue"].get("values", [])
        return [unwrap_attribute_value(v) for v in values]
    if "kvlistValue" in attr_value:
        entries = attr_value["kvlistValue"].get("values", [])
        return {
            entry["key"]: unwrap_attribute_value(entry["value"])
            for entry in entries
            if "key" in entry and "value" in entry
        }

    # Already a plain value (shouldn't happen in well-formed OTLP)
    return attr_value


def get_attr(span: dict, key: str) -> Any | None:
    """Look up a single attribute by key from an OTLP span."""
    for attr in span.get("attributes", []):
        if attr.get("key") == key:
            return unwrap_attribute_value(attr["value"])
    return None


# ── Span → Step conversion ──────────────────────────────────────────

def infer_step_kind(span: dict) -> StepKind:
    """Determine the :class:`StepKind` for an OTLP span.

    Resolution order:
    1. ``gen_ai.operation.name`` attribute (primary)
    2. Span name substring heuristic (fallback)
    3. ``StepKind.CUSTOM`` (default)
    """
    operation = get_attr(span, "gen_ai.operation.name")
    if operation and operation in _OPERATION_TO_KIND:
        return _OPERATION_TO_KIND[operation]

    # Fallback: match span name
    span_name = span.get("name", "").lower()
    for hint, kind in _NAME_HINTS:
        if hint in span_name:
            return kind

    return StepKind.CUSTOM


def _resolve_agent_name(
    span: dict,
    parent_span: dict | None,
    default_agent_name: str,
) -> str:
    """Resolve the agent name for a span.

    1. ``gen_ai.agent.name`` on the span itself
    2. ``gen_ai.agent.name`` on the parent span (inherited context)
    3. *default_agent_name*
    """
    name = get_attr(span, "gen_ai.agent.name")
    if name:
        return name
    if parent_span is not None:
        name = get_attr(parent_span, "gen_ai.agent.name")
        if name:
            return name
    return default_agent_name


def _extract_model(span: dict) -> str | None:
    """Extract the LLM model name from gen_ai or legacy attributes."""
    return (
        get_attr(span, "gen_ai.request.model")
        or get_attr(span, "gen_ai.response.model")
        or get_attr(span, "llm.request.model")  # OpenLLMetry legacy
    )


def _extract_tokens(span: dict) -> tuple[int | None, int | None]:
    """Extract prompt / completion token counts."""
    prompt = (
        get_attr(span, "gen_ai.usage.input_tokens")
        or get_attr(span, "gen_ai.usage.prompt_tokens")
        or get_attr(span, "llm.usage.prompt_tokens")  # legacy
    )
    completion = (
        get_attr(span, "gen_ai.usage.output_tokens")
        or get_attr(span, "gen_ai.usage.completion_tokens")
        or get_attr(span, "llm.usage.completion_tokens")  # legacy
    )
    return (
        int(prompt) if prompt is not None else None,
        int(completion) if completion is not None else None,
    )


def _extract_tool_info(span: dict) -> tuple[str | None, dict | None, Any, bool | None]:
    """Extract tool name, args, result, and success from a tool span.

    Returns (tool_name, tool_args, tool_result, tool_succeeded).
    """
    tool_name = (
        get_attr(span, "gen_ai.tool.name")
        or get_attr(span, "tool.name")
        or get_attr(span, "traceloop.entity.name")  # OpenLLMetry legacy
    )
    tool_args = get_attr(span, "gen_ai.tool.call.arguments")
    tool_result = get_attr(span, "gen_ai.tool.call.result")

    # Determine success from span status
    status = span.get("status", {})
    status_code = status.get("code") if isinstance(status, dict) else None
    if status_code == 2:  # ERROR
        succeeded = False
    elif tool_result is not None or status_code == 1:  # OK
        succeeded = True
    else:
        succeeded = None

    return tool_name, tool_args, tool_result, succeeded


def _nano_to_iso(nano: str | int | None) -> str | None:
    """Convert nanosecond Unix timestamp to ISO-8601 string (or None)."""
    if nano is None:
        return None
    try:
        ts_ns = int(nano)
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)
        return dt.isoformat()
    except (ValueError, TypeError, OSError):
        return None


def span_to_step(
    span: dict,
    parent_span: dict | None,
    default_agent_name: str,
) -> Step:
    """Convert a single OTLP span dict to a canonical :class:`Step`."""
    from datetime import datetime

    kind = infer_step_kind(span)
    agent_name = _resolve_agent_name(span, parent_span, default_agent_name)
    agent_id = span.get("spanId", "")

    # Content: prefer gen_ai.content.prompt or completion, fall back to span name
    content = (
        get_attr(span, "gen_ai.content.completion")
        or get_attr(span, "gen_ai.content.prompt")
    )

    # Timestamp
    ts_iso = _nano_to_iso(span.get("startTimeUnixNano"))
    timestamp = None
    if ts_iso:
        try:
            timestamp = datetime.fromisoformat(ts_iso)
        except (ValueError, TypeError):
            pass

    # Kind-specific fields
    model = None
    prompt_tokens = None
    completion_tokens = None
    tool_name = None
    tool_args = None
    tool_result = None
    tool_succeeded = None

    if kind == StepKind.LLM_CALL:
        model = _extract_model(span)
        prompt_tokens, completion_tokens = _extract_tokens(span)
    elif kind == StepKind.TOOL_CALL:
        tool_name, tool_args, tool_result, tool_succeeded = _extract_tool_info(span)

    # Build metadata
    metadata: dict[str, Any] = {}
    operation = get_attr(span, "gen_ai.operation.name")
    if operation:
        metadata["operation"] = operation
    provider = get_attr(span, "gen_ai.system") or get_attr(span, "gen_ai.provider.name")
    if provider:
        metadata["provider"] = provider

    return Step(
        kind=kind,
        agent_id=agent_id,
        agent_name=agent_name,
        content=content,
        timestamp=timestamp,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
        tool_succeeded=tool_succeeded,
        metadata=metadata,
    )
