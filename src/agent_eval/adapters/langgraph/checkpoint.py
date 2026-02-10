"""LangGraph checkpoint reader.

Reads checkpoint snapshots from a LangGraph trace and converts them
into Episode step sequences.  Checkpoints capture the full message
history at each state transition, providing an alternative to
event-based reconstruction.

Checkpoint format (from LangGraph's ``MemorySaver`` / ``SqliteSaver``)::

    {
        "checkpoint_id": "cp-001",
        "timestamp": "2025-01-15T10:00:00Z",
        "channel_values": {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "...", "tool_calls": [...]},
                {"role": "tool", "content": "...", "name": "..."},
            ]
        },
        "metadata": {"step": 0}
    }
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_eval.core.exceptions import AdapterError
from agent_eval.core.models import Episode, Step, StepKind


def load_checkpoints(source: str) -> list[dict[str, Any]]:
    """Load checkpoints from a JSON trace file.

    Parameters
    ----------
    source
        Path to a JSON file with a ``checkpoints`` key.

    Returns
    -------
    list[dict]
        The raw checkpoint list.
    """
    p = Path(source)
    if not p.is_file():
        raise AdapterError(f"Checkpoint file not found: {source}")
    try:
        content = p.read_text(encoding="utf-8")
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"Invalid JSON in {source}: {exc}") from exc

    checkpoints = data.get("checkpoints", [])
    if not checkpoints:
        raise AdapterError(f"No 'checkpoints' key found in {source}")
    return checkpoints


def checkpoint_to_episode(
    checkpoint: dict[str, Any],
    episode_id: str | None = None,
) -> Episode:
    """Convert a single checkpoint snapshot into an Episode.

    The checkpoint's ``channel_values.messages`` list is converted
    into Steps, preserving role → StepKind mapping:

    - ``user`` → MESSAGE
    - ``assistant`` → MESSAGE (or TOOL_CALL if tool_calls present)
    - ``tool`` → TOOL_RESULT
    - ``system`` → MESSAGE

    Parameters
    ----------
    checkpoint
        A single checkpoint dict.
    episode_id
        Optional custom episode ID.  Defaults to checkpoint_id.
    """
    cp_id = checkpoint.get("checkpoint_id", str(uuid.uuid4()))
    ts_str = checkpoint.get("timestamp")
    channel_values = checkpoint.get("channel_values", {})
    messages = channel_values.get("messages", [])
    cp_metadata = checkpoint.get("metadata", {})

    ts = _parse_ts(ts_str)
    steps = _messages_to_steps(messages, timestamp=ts)

    # Final answer: last assistant message content
    final_answer = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content:
                final_answer = content
                break

    return Episode(
        episode_id=episode_id or cp_id,
        steps=steps,
        source_framework="langgraph",
        task_description=_extract_task(messages),
        final_answer=final_answer,
        started_at=ts,
        ended_at=ts,
        metadata={"checkpoint_id": cp_id, **cp_metadata},
    )


def checkpoints_to_episodes(
    checkpoints: list[dict[str, Any]],
) -> list[Episode]:
    """Convert each checkpoint into a separate Episode.

    Useful for examining the state at each step of graph execution.
    """
    return [
        checkpoint_to_episode(cp, episode_id=cp.get("checkpoint_id"))
        for cp in checkpoints
    ]


def latest_checkpoint_to_episode(
    checkpoints: list[dict[str, Any]],
) -> Episode:
    """Convert the final checkpoint (most complete state) to an Episode.

    Parameters
    ----------
    checkpoints
        Ordered list of checkpoints (earliest first).
    """
    if not checkpoints:
        raise AdapterError("Empty checkpoint list")
    return checkpoint_to_episode(checkpoints[-1])


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _messages_to_steps(
    messages: list[dict[str, Any]],
    timestamp: datetime | None = None,
) -> list[Step]:
    steps: list[Step] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        tool_name = msg.get("name")

        if role == "user":
            steps.append(
                Step(
                    kind=StepKind.MESSAGE,
                    agent_id="user",
                    agent_name="user",
                    content=content,
                    timestamp=timestamp,
                )
            )

        elif role == "assistant":
            if tool_calls:
                for tc in tool_calls:
                    tc_name = (
                        tc.get("name", "unknown") if isinstance(tc, dict) else "unknown"
                    )
                    tc_args = tc.get("args") if isinstance(tc, dict) else None
                    steps.append(
                        Step(
                            kind=StepKind.TOOL_CALL,
                            agent_id="assistant",
                            agent_name="assistant",
                            tool_name=tc_name,
                            tool_args=tc_args if isinstance(tc_args, dict) else None,
                            timestamp=timestamp,
                        )
                    )
            if content:
                steps.append(
                    Step(
                        kind=StepKind.MESSAGE,
                        agent_id="assistant",
                        agent_name="assistant",
                        content=content,
                        timestamp=timestamp,
                    )
                )

        elif role == "tool":
            steps.append(
                Step(
                    kind=StepKind.TOOL_RESULT,
                    agent_id="tool",
                    agent_name=tool_name or "tool",
                    tool_name=tool_name,
                    tool_result=content,
                    tool_succeeded=True,
                    timestamp=timestamp,
                )
            )

        elif role == "system":
            steps.append(
                Step(
                    kind=StepKind.MESSAGE,
                    agent_id="system",
                    agent_name="system",
                    content=content,
                    timestamp=timestamp,
                )
            )

    return steps


def _extract_task(messages: list[dict[str, Any]]) -> str | None:
    """Extract task from first user message."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content")
    return None


def _parse_ts(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
