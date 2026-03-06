"""AutoGen adapter: converts event.txt / JSONL / JSON logs into canonical Episodes.

Ported and generalized from log_evaluator.py:_extract_metrics.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_eval.adapters.autogen.event_parser import parse_events
from agent_eval.adapters.autogen.tool_failure import is_tool_call_failed
from agent_eval.core.exceptions import AdapterError
from agent_eval.core.models import Episode, Step, StepKind

# File names recognised when resolving a directory to a log file.
_LOG_FILE_NAMES = ("event.txt", "events.jsonl", "events.json")

# Glob patterns used by load_episodes to discover log files.
_LOG_GLOB_PATTERNS = ("event.txt", "events.jsonl", "events.json")

# Regex to strip AutoGen's universal ``_<uuid>/<uuid>`` agent ID suffix.
_UUID_SUFFIX_RE = re.compile(
    r"_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"(?:/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?$"
)

# Known AutoGen framework-internal agent class names.  These agents are
# plumbing — they handle routing and orchestration inside group chats and
# should not be scored as worker agents.
AUTOGEN_FRAMEWORK_AGENTS: frozenset[str] = frozenset(
    {
        "SelectorGroupChatManager",
        "RoundRobinGroupChatManager",
        "MagenticOneOrchestrator",
    }
)


def strip_uuid_suffix(name: str) -> str:
    """Remove the ``_<uuid>/<uuid>`` suffix added by AutoGen's runtime."""
    return _UUID_SUFFIX_RE.sub("", name)


class AutoGenAdapter:
    """Convert AutoGen event logs into canonical Episodes.

    Supports three log formats (auto-detected):
    - ``event.txt`` — text + multi-line JSON (original AutoGen format)
    - ``events.jsonl`` — one JSON object per line (structured event logger)
    - ``events.json`` — JSON array of event objects

    Parameters
    ----------
    agent_names
        Known agent names for activity detection.  Agent names in log
        senders may have suffixes (e.g. ``FinanceExpert_abc123``);
        the adapter matches by substring.
    orchestrator_name
        The orchestrator agent name to exclude from per-agent turn counts.
    answer_patterns
        Ordered list of marker strings used to locate the final answer in
        the raw log content.  The first match wins.  Defaults to
        ``["<ANSWER>:", "FINAL ANSWER:", "FINAL_ANSWER"]``.
    strip_uuids
        If ``True`` (the default), strip AutoGen's ``_<uuid>/<uuid>``
        suffix from agent names so that
        ``FinanceExpert_abc123-.../.../`` becomes ``FinanceExpert``.
    framework_agents
        Set of agent class names to tag as framework-internal.  These
        agents are still included in the Episode but their steps carry
        ``metadata["framework_agent"] = True`` so scorers can filter
        them.  Defaults to :data:`AUTOGEN_FRAMEWORK_AGENTS`.
    """

    # ----- dispatch table: event "type" → handler method name -----
    _EVENT_HANDLERS: dict[str, str] = {
        "Message": "_handle_message",
        "MessageEvent": "_handle_message_event",
        "MessageDroppedEvent": "_handle_message_dropped",
        "MessageHandlerExceptionEvent": "_handle_message_handler_exception",
        "ToolCall": "_handle_tool_call",
        "ToolCallEvent": "_handle_tool_call_event",
        "LLMCall": "_handle_llm_call",
        "LLMStreamStartEvent": "_handle_llm_stream_start",
        "LLMStreamEndEvent": "_handle_llm_stream_end",
        "FactCheckResult": "_handle_fact_check",
        "AgentConstructionExceptionEvent": "_handle_agent_construction_exception",
    }

    def __init__(
        self,
        agent_names: list[str] | None = None,
        orchestrator_name: str = "SalesNegotiator",
        answer_patterns: list[str] | None = None,
        strip_uuids: bool = True,
        framework_agents: frozenset[str] | None = None,
    ) -> None:
        self.agent_names = agent_names or []
        self.orchestrator_name = orchestrator_name
        self.answer_patterns = answer_patterns or [
            "<ANSWER>:",
            "FINAL ANSWER:",
            "FINAL_ANSWER",
        ]
        self.strip_uuids = strip_uuids
        self.framework_agents = (
            framework_agents
            if framework_agents is not None
            else AUTOGEN_FRAMEWORK_AGENTS
        )

    @property
    def framework_name(self) -> str:
        return "autogen"

    def load_episode(self, source: str, **kwargs: Any) -> Episode:
        """Load a single Episode from a log file or directory.

        Parameters
        ----------
        source
            Path to an event log file (``event.txt``, ``events.jsonl``,
            ``events.json``) or a directory containing one.
        """
        log_file = self._resolve_log_path(source)
        content = log_file.read_text(encoding="utf-8", errors="ignore")
        raw_events = parse_events(content)
        return self._build_episode(
            raw_events=raw_events,
            content=content,
            source_path=str(log_file),
        )

    def load_episodes(self, source: str, **kwargs: Any) -> list[Episode]:
        """Load Episodes from a directory tree.

        Finds all recognised log files (``event.txt``, ``events.jsonl``,
        ``events.json``) under *source* recursively and converts each into
        an Episode.  Episodes with zero steps are skipped (avoids picking
        up non-AutoGen JSON files).
        """
        base = Path(source)
        if not base.is_dir():
            return [self.load_episode(source)]

        seen_dirs: set[Path] = set()
        episodes: list[Episode] = []

        for pattern in _LOG_GLOB_PATTERNS:
            for event_file in sorted(base.rglob(pattern)):
                # Avoid processing the same directory twice if it contains
                # multiple recognised file names.
                if event_file.parent in seen_dirs:
                    continue
                seen_dirs.add(event_file.parent)
                try:
                    content = event_file.read_text(
                        encoding="utf-8", errors="ignore"
                    )
                    raw_events = parse_events(content)
                    ep = self._build_episode(
                        raw_events, content, str(event_file)
                    )
                    if ep.steps:
                        episodes.append(ep)
                except Exception:
                    continue
        return episodes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_log_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_file():
            return p
        if p.is_dir():
            for name in _LOG_FILE_NAMES:
                candidate = p / name
                if candidate.exists():
                    return candidate
        raise AdapterError(
            f"Could not find AutoGen log file at: {path}"
        )

    def _build_episode(
        self,
        raw_events: list[dict],
        content: str,
        source_path: str,
    ) -> Episode:
        steps = self._events_to_steps(raw_events)

        # Tag framework agent steps
        for step in steps:
            if step.agent_name and self._is_framework_agent(step.agent_name):
                step.metadata["framework_agent"] = True

        # Determine timestamps
        started_at, ended_at = self._extract_timestamps(raw_events, content)

        # Extract final answer from raw text
        final_answer = self._extract_final_answer(content)

        return Episode(
            episode_id=str(uuid.uuid4()),
            steps=steps,
            source_framework="autogen",
            task_description=None,
            final_answer=final_answer,
            started_at=started_at,
            ended_at=ended_at,
            metadata={"source_path": source_path},
        )

    def _events_to_steps(self, raw_events: list[dict]) -> list[Step]:
        steps: list[Step] = []

        for event in raw_events:
            event_type = event.get("type", "")
            handler_name = self._EVENT_HANDLERS.get(event_type)
            if handler_name is None:
                continue
            handler = getattr(self, handler_name)
            step = handler(event)
            if step is not None:
                steps.append(step)

        return steps

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_message(self, event: dict) -> Step:
        sender = event.get("sender", "")
        agent_name = self._resolve_agent_name(sender)
        return Step(
            kind=StepKind.MESSAGE,
            agent_id=sender,
            agent_name=agent_name,
            content=event.get("content"),
            timestamp=self._parse_timestamp(event.get("timestamp")),
            metadata={"raw_event": event},
        )

    def _handle_message_event(self, event: dict) -> Step:
        sender = event.get("sender", "")
        agent_name = self._resolve_agent_name(sender)
        payload = event.get("payload", "")
        content = payload if isinstance(payload, str) else str(payload)
        meta: dict[str, Any] = {"raw_event": event}
        for key in ("receiver", "kind", "delivery_stage"):
            if key in event:
                meta[key] = event[key]
        return Step(
            kind=StepKind.MESSAGE,
            agent_id=sender,
            agent_name=agent_name,
            content=content,
            timestamp=self._parse_timestamp(event.get("timestamp")),
            metadata=meta,
        )

    def _handle_message_dropped(self, event: dict) -> Step:
        sender = event.get("sender", "")
        meta: dict[str, Any] = {
            "custom_type": "MessageDroppedEvent",
            "raw_event": event,
        }
        for key in ("receiver", "reason", "payload"):
            if key in event:
                meta[key] = event[key]
        return Step(
            kind=StepKind.CUSTOM,
            agent_id=sender,
            agent_name=self._resolve_agent_name(sender),
            content=event.get("reason"),
            timestamp=self._parse_timestamp(event.get("timestamp")),
            metadata=meta,
        )

    def _handle_message_handler_exception(self, event: dict) -> Step:
        meta: dict[str, Any] = {
            "custom_type": "MessageHandlerExceptionEvent",
            "raw_event": event,
        }
        for key in ("handler_class", "exception", "traceback"):
            if key in event:
                meta[key] = event[key]
        return Step(
            kind=StepKind.CUSTOM,
            agent_id=event.get("handler_class", ""),
            agent_name="",
            content=event.get("exception"),
            timestamp=self._parse_timestamp(event.get("timestamp")),
            metadata=meta,
        )

    def _handle_tool_call(self, event: dict) -> Step:
        tool_name = event.get("tool") or event.get("tool_name", "unknown")
        result = event.get("result", "")
        succeeded = not is_tool_call_failed(result)
        agent_id = event.get("agent_id", "")
        return Step(
            kind=StepKind.TOOL_CALL,
            agent_id=agent_id,
            agent_name=self._resolve_agent_name(agent_id),
            tool_name=tool_name,
            tool_args=event.get("arguments"),
            tool_result=result,
            tool_succeeded=succeeded,
            timestamp=self._parse_timestamp(event.get("timestamp")),
            metadata={"raw_event": event},
        )

    def _handle_tool_call_event(self, event: dict) -> Step:
        tool_name = event.get("tool_name") or event.get("tool", "unknown")
        result = event.get("result", "")
        succeeded = not is_tool_call_failed(result)
        agent_id = event.get("agent_id", "")
        return Step(
            kind=StepKind.TOOL_CALL,
            agent_id=agent_id,
            agent_name=self._resolve_agent_name(agent_id),
            tool_name=tool_name,
            tool_args=event.get("arguments"),
            tool_result=result,
            tool_succeeded=succeeded,
            timestamp=self._parse_timestamp(event.get("timestamp")),
            metadata={"raw_event": event},
        )

    def _handle_llm_call(self, event: dict) -> Step:
        return Step(
            kind=StepKind.LLM_CALL,
            agent_id="",
            agent_name="",
            model=event.get("model"),
            prompt_tokens=event.get("prompt_tokens"),
            completion_tokens=event.get("completion_tokens"),
            timestamp=self._parse_timestamp(event.get("timestamp")),
            metadata={"raw_event": event},
        )

    def _handle_llm_stream_start(self, event: dict) -> Step:
        agent_id = event.get("agent_id", "")
        return Step(
            kind=StepKind.LLM_CALL,
            agent_id=agent_id,
            agent_name=self._resolve_agent_name(agent_id),
            model=event.get("model"),
            timestamp=self._parse_timestamp(event.get("timestamp")),
            metadata={
                "stream_phase": "start",
                "raw_event": event,
            },
        )

    def _handle_llm_stream_end(self, event: dict) -> Step:
        agent_id = event.get("agent_id", "")
        return Step(
            kind=StepKind.LLM_CALL,
            agent_id=agent_id,
            agent_name=self._resolve_agent_name(agent_id),
            model=event.get("model"),
            prompt_tokens=event.get("prompt_tokens"),
            completion_tokens=event.get("completion_tokens"),
            timestamp=self._parse_timestamp(event.get("timestamp")),
            metadata={
                "stream_phase": "end",
                "raw_event": event,
            },
        )

    def _handle_fact_check(self, event: dict) -> Step:
        return Step(
            kind=StepKind.FACT_CHECK,
            agent_id=event.get("agent_name", ""),
            agent_name=event.get("agent_name", ""),
            content=event.get("answer_preview"),
            timestamp=self._parse_timestamp(event.get("timestamp")),
            metadata={
                "verdict": event.get("verdict"),
                "reasoning": event.get("reasoning", []),
                "question": event.get("question"),
                "raw_event": event,
            },
        )

    def _handle_agent_construction_exception(self, event: dict) -> Step:
        return Step(
            kind=StepKind.CUSTOM,
            agent_id=event.get("agent_class", ""),
            agent_name="",
            content=event.get("exception"),
            timestamp=self._parse_timestamp(event.get("timestamp")),
            metadata={
                "custom_type": "AgentConstructionExceptionEvent",
                "agent_class": event.get("agent_class"),
                "raw_event": event,
            },
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _resolve_agent_name(self, sender: str | None) -> str:
        """Match a sender string to a known agent name.

        Resolution order:
        1. If *sender* matches a known ``agent_names`` entry (substring), use it.
        2. If ``strip_uuids`` is enabled, strip the ``_<uuid>/<uuid>`` suffix.
        3. Otherwise return *sender* as-is.
        """
        if not sender:
            return sender or ""
        for name in self.agent_names:
            if name in sender:
                return name
        if self.strip_uuids:
            return strip_uuid_suffix(sender)
        return sender

    def _is_framework_agent(self, agent_name: str) -> bool:
        """Return ``True`` if *agent_name* matches a known framework agent."""
        clean = strip_uuid_suffix(agent_name) if self.strip_uuids else agent_name
        if clean in self.framework_agents:
            return True
        if self.orchestrator_name and clean == self.orchestrator_name:
            return True
        return False

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    def _extract_timestamps(
        self,
        raw_events: list[dict],
        content: str,
    ) -> tuple[datetime | None, datetime | None]:
        """Extract start/end timestamps from events or log text."""
        timestamps = [
            datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
            for e in raw_events
            if "timestamp" in e and isinstance(e["timestamp"], str)
        ]

        if len(timestamps) >= 2:
            return timestamps[0], timestamps[-1]

        # Fallback: parse from log line timestamps
        ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
        text_timestamps: list[datetime] = []
        for line in content.split("\n"):
            match = ts_pattern.search(line)
            if match:
                try:
                    ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                    text_timestamps.append(ts)
                except ValueError:
                    pass

        if len(text_timestamps) >= 2:
            return text_timestamps[0], text_timestamps[-1]

        return None, None

    def _extract_final_answer(self, content: str) -> str | None:
        """Extract the final answer from log content.

        Tries each marker in ``self.answer_patterns`` in order. The first
        marker found wins.  For ``<ANSWER>:`` specifically, the method also
        looks for a ``</ANSWER>`` closing tag or an em-dash separator.
        """
        for pattern in self.answer_patterns:
            idx = content.find(pattern)
            if idx < 0:
                continue

            # For the XML-style marker, try to find a closing tag
            if pattern == "<ANSWER>:":
                answer_end = content.find("</ANSWER>", idx)
                if answer_end < 0:
                    answer_end = content.find("\u2500" * 16, idx)
                    if answer_end < 0:
                        answer_end = len(content)
                return content[idx + len(pattern) : answer_end].strip()

            # For other patterns, take everything from the marker to the
            # next double-newline or end of content (capped at 5000 chars).
            start = idx + len(pattern)
            end = content.find("\n\n", start)
            if end < 0 or end - start > 5000:
                end = min(start + 5000, len(content))
            answer = content[start:end].strip()
            if answer:
                return answer

        return None
