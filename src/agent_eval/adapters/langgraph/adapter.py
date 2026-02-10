"""LangGraph adapter: converts LangGraph traces into canonical Episodes.

Supports two input formats:
1. **astream_events JSON** — the ``events`` list produced by
   ``graph.astream_events()``, each with ``event``, ``name``, ``data``.
2. **Checkpoint JSON** — optional ``checkpoints`` list with message
   history snapshots (see ``checkpoint.py``).

The adapter maps LangGraph event types to canonical Step kinds:

| LangGraph Event          | Step Kind  |
|--------------------------|------------|
| ``on_chain_start/end``   | MESSAGE    |
| ``on_chat_model_start``  | LLM_CALL   |
| ``on_chat_model_end``    | LLM_CALL   |
| ``on_tool_start``        | TOOL_CALL  |
| ``on_tool_end``          | TOOL_RESULT|
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_eval.core.exceptions import AdapterError
from agent_eval.core.models import Episode, Step, StepKind


class LangGraphAdapter:
    """Convert LangGraph traces into canonical Episodes.

    Parameters
    ----------
    default_agent_name
        Agent name to assign when the event doesn't specify one.
    """

    def __init__(self, default_agent_name: str = "agent") -> None:
        self.default_agent_name = default_agent_name

    @property
    def framework_name(self) -> str:
        return "langgraph"

    def load_episode(self, source: str, **kwargs: Any) -> Episode:
        """Load a single Episode from a JSON trace file or dict.

        Parameters
        ----------
        source
            Path to a ``.json`` file containing a LangGraph trace with
            an ``events`` key.
        """
        trace = self._load_trace(source)
        return self._trace_to_episode(trace, source_path=source)

    def load_episodes(self, source: str, **kwargs: Any) -> list[Episode]:
        """Load Episodes from a directory of JSON trace files.

        Finds all ``.json`` files under *source* (recursively) and
        converts each into an Episode.
        """
        base = Path(source)
        if not base.is_dir():
            return [self.load_episode(source)]

        episodes: list[Episode] = []
        for json_file in sorted(base.rglob("*.json")):
            try:
                ep = self.load_episode(str(json_file))
                episodes.append(ep)
            except Exception:
                continue
        return episodes

    def load_from_dict(self, trace: dict[str, Any]) -> Episode:
        """Load an Episode directly from a trace dictionary."""
        return self._trace_to_episode(trace, source_path="<dict>")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_trace(self, source: str) -> dict[str, Any]:
        p = Path(source)
        if not p.is_file():
            raise AdapterError(f"Trace file not found: {source}")
        try:
            content = p.read_text(encoding="utf-8")
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"Invalid JSON in {source}: {exc}") from exc

    def _trace_to_episode(self, trace: dict[str, Any], source_path: str) -> Episode:
        events = trace.get("events", [])
        if not events:
            raise AdapterError("Trace has no 'events' key or events list is empty")

        steps = self._events_to_steps(events)

        # Extract timestamps
        started_at, ended_at = self._extract_timestamps(events)

        # Extract task description from first chain_start input
        task_description = self._extract_task(events)

        # Extract final answer from last chain_end output
        final_answer = self._extract_final_answer(events)

        # Thread / run metadata
        thread_id = trace.get("thread_id", "")
        trace_metadata = trace.get("metadata", {})

        return Episode(
            episode_id=thread_id or str(uuid.uuid4()),
            steps=steps,
            source_framework="langgraph",
            task_description=task_description,
            final_answer=final_answer,
            started_at=started_at,
            ended_at=ended_at,
            metadata={
                "source_path": source_path,
                "thread_id": thread_id,
                **trace_metadata,
            },
        )

    def _events_to_steps(self, events: list[dict]) -> list[Step]:
        steps: list[Step] = []
        # Track tool start events to pair with tool end
        tool_starts: dict[str, dict] = {}

        for event in events:
            event_type = event.get("event", event.get("type", ""))
            name = event.get("name", "")
            data = event.get("data", {})
            ts = self._parse_timestamp(event.get("timestamp"))
            run_id = event.get("run_id", "")
            event_meta = event.get("metadata", {})

            if event_type == "on_chain_start":
                # Extract user message from input
                content = self._extract_message_content(data.get("input"))
                if content:
                    steps.append(
                        Step(
                            kind=StepKind.MESSAGE,
                            agent_id=run_id,
                            agent_name=name or self.default_agent_name,
                            content=content,
                            timestamp=ts,
                            metadata={"event_type": event_type},
                        )
                    )

            elif event_type == "on_chain_end":
                content = self._extract_message_content(data.get("output"))
                if content:
                    steps.append(
                        Step(
                            kind=StepKind.MESSAGE,
                            agent_id=run_id,
                            agent_name=name or self.default_agent_name,
                            content=content,
                            timestamp=ts,
                            metadata={"event_type": event_type},
                        )
                    )

            elif event_type in ("on_chat_model_start", "on_chat_model_end"):
                model_name = event_meta.get("ls_model_name", name)

                prompt_tokens = None
                completion_tokens = None
                content = None

                if event_type == "on_chat_model_end":
                    output = data.get("output", {})
                    if isinstance(output, dict):
                        content = output.get("content")
                        usage = data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")

                steps.append(
                    Step(
                        kind=StepKind.LLM_CALL,
                        agent_id=run_id,
                        agent_name=model_name,
                        content=content,
                        model=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        timestamp=ts,
                        metadata={"event_type": event_type},
                    )
                )

            elif event_type == "on_tool_start":
                tool_starts[run_id] = event
                tool_input = data.get("input", {})
                steps.append(
                    Step(
                        kind=StepKind.TOOL_CALL,
                        agent_id=run_id,
                        agent_name=self.default_agent_name,
                        tool_name=name,
                        tool_args=tool_input if isinstance(tool_input, dict) else None,
                        content=str(tool_input)
                        if not isinstance(tool_input, dict)
                        else None,
                        timestamp=ts,
                        metadata={"event_type": event_type},
                    )
                )

            elif event_type == "on_tool_end":
                output = data.get("output", "")
                # Determine success: no explicit error markers
                succeeded = not self._looks_like_error(output)
                steps.append(
                    Step(
                        kind=StepKind.TOOL_RESULT,
                        agent_id=run_id,
                        agent_name=self.default_agent_name,
                        tool_name=name,
                        tool_result=output,
                        tool_succeeded=succeeded,
                        timestamp=ts,
                        metadata={"event_type": event_type},
                    )
                )

        return steps

    def _extract_message_content(self, data: Any) -> str | None:
        """Extract human-readable content from LangGraph event data."""
        if data is None:
            return None

        if isinstance(data, str):
            return data if data.strip() else None

        if isinstance(data, dict):
            # Check for messages list
            messages = data.get("messages", [])
            if messages:
                # Messages can be a list of dicts or list of lists of dicts
                last_msg = messages[-1] if messages else None
                if isinstance(last_msg, list) and last_msg:
                    last_msg = last_msg[-1]
                if isinstance(last_msg, dict):
                    content = last_msg.get("content", "")
                    return content if content else None

            # Fallback: direct content key
            content = data.get("content")
            if content:
                return content

        return None

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    def _extract_timestamps(
        self, events: list[dict]
    ) -> tuple[datetime | None, datetime | None]:
        timestamps = []
        for e in events:
            ts = self._parse_timestamp(e.get("timestamp"))
            if ts:
                timestamps.append(ts)
        if len(timestamps) >= 2:
            return timestamps[0], timestamps[-1]
        if timestamps:
            return timestamps[0], None
        return None, None

    def _extract_task(self, events: list[dict]) -> str | None:
        """Extract task description from first on_chain_start."""
        for e in events:
            if e.get("event", e.get("type", "")) == "on_chain_start":
                data = e.get("data", {})
                inp = data.get("input")
                if isinstance(inp, str):
                    return inp
                if isinstance(inp, dict):
                    messages = inp.get("messages", [])
                    if messages:
                        first = messages[0]
                        if isinstance(first, dict):
                            return first.get("content")
                return None
        return None

    def _extract_final_answer(self, events: list[dict]) -> str | None:
        """Extract final answer from last on_chain_end."""
        for e in reversed(events):
            if e.get("event", e.get("type", "")) == "on_chain_end":
                data = e.get("data", {})
                content = self._extract_message_content(data.get("output"))
                return content
        return None

    @staticmethod
    def _looks_like_error(output: Any) -> bool:
        """Simple heuristic for tool failure detection."""
        if isinstance(output, dict):
            for key in ("error", "errors", "exception", "traceback"):
                if key in output:
                    return True
            if output.get("success") is False or output.get("ok") is False:
                return True
        if isinstance(output, str):
            lower = output.lower()
            for pattern in ("error", "exception", "traceback", "failed"):
                if pattern in lower:
                    return True
        return False
