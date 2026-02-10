"""AutoGen adapter: converts event.txt logs into canonical Episodes.

Ported and generalized from log_evaluator.py:_extract_metrics.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_eval.adapters.autogen.event_parser import extract_json_events
from agent_eval.adapters.autogen.tool_failure import is_tool_call_failed
from agent_eval.core.exceptions import AdapterError
from agent_eval.core.models import Episode, Step, StepKind


class AutoGenAdapter:
    """Convert AutoGen event.txt logs into canonical Episodes.

    Parameters
    ----------
    agent_names
        Known agent names for activity detection.  Agent names in log
        senders may have suffixes (e.g. ``FinanceExpert_abc123``);
        the adapter matches by substring.
    orchestrator_name
        The orchestrator agent name to exclude from per-agent turn counts.
    """

    def __init__(
        self,
        agent_names: list[str] | None = None,
        orchestrator_name: str = "SalesNegotiator",
    ) -> None:
        self.agent_names = agent_names or []
        self.orchestrator_name = orchestrator_name

    @property
    def framework_name(self) -> str:
        return "autogen"

    def load_episode(self, source: str, **kwargs: Any) -> Episode:
        """Load a single Episode from an event.txt file or directory.

        Parameters
        ----------
        source
            Path to an ``event.txt`` file or a directory containing one.
        """
        log_file = self._resolve_log_path(source)
        content = log_file.read_text(encoding="utf-8", errors="ignore")
        raw_events = extract_json_events(content)
        return self._build_episode(
            raw_events=raw_events,
            content=content,
            source_path=str(log_file),
        )

    def load_episodes(self, source: str, **kwargs: Any) -> list[Episode]:
        """Load Episodes from a directory tree.

        Finds all ``event.txt`` files under *source* (recursively) and
        converts each into an Episode.
        """
        base = Path(source)
        if not base.is_dir():
            return [self.load_episode(source)]

        episodes: list[Episode] = []
        for event_file in sorted(base.rglob("event.txt")):
            try:
                content = event_file.read_text(encoding="utf-8", errors="ignore")
                raw_events = extract_json_events(content)
                ep = self._build_episode(raw_events, content, str(event_file))
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
            event_file = p / "event.txt"
            if event_file.exists():
                return event_file
        raise AdapterError(f"Could not find event.txt at: {path}")

    def _build_episode(
        self,
        raw_events: list[dict],
        content: str,
        source_path: str,
    ) -> Episode:
        steps = self._events_to_steps(raw_events)

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

            if event_type == "Message":
                sender = event.get("sender", "")
                agent_name = self._resolve_agent_name(sender)
                ts = self._parse_timestamp(event.get("timestamp"))
                steps.append(
                    Step(
                        kind=StepKind.MESSAGE,
                        agent_id=sender,
                        agent_name=agent_name,
                        content=event.get("content"),
                        timestamp=ts,
                        metadata={"raw_event": event},
                    )
                )

            elif event_type == "ToolCall":
                tool_name = event.get("tool") or event.get("tool_name", "unknown")
                result = event.get("result", "")
                succeeded = not is_tool_call_failed(result)
                ts = self._parse_timestamp(event.get("timestamp"))
                steps.append(
                    Step(
                        kind=StepKind.TOOL_CALL,
                        agent_id="",
                        agent_name="",
                        tool_name=tool_name,
                        tool_args=event.get("arguments"),
                        tool_result=result,
                        tool_succeeded=succeeded,
                        timestamp=ts,
                        metadata={"raw_event": event},
                    )
                )

            elif event_type == "LLMCall":
                ts = self._parse_timestamp(event.get("timestamp"))
                steps.append(
                    Step(
                        kind=StepKind.LLM_CALL,
                        agent_id="",
                        agent_name="",
                        model=event.get("model"),
                        prompt_tokens=event.get("prompt_tokens"),
                        completion_tokens=event.get("completion_tokens"),
                        timestamp=ts,
                        metadata={"raw_event": event},
                    )
                )

            elif event_type == "FactCheckResult":
                ts = self._parse_timestamp(event.get("timestamp"))
                steps.append(
                    Step(
                        kind=StepKind.FACT_CHECK,
                        agent_id=event.get("agent_name", ""),
                        agent_name=event.get("agent_name", ""),
                        content=event.get("answer_preview"),
                        timestamp=ts,
                        metadata={
                            "verdict": event.get("verdict"),
                            "reasoning": event.get("reasoning", []),
                            "question": event.get("question"),
                            "raw_event": event,
                        },
                    )
                )

        return steps

    def _resolve_agent_name(self, sender: str) -> str:
        """Match a sender string to a known agent name."""
        for name in self.agent_names:
            if name in sender:
                return name
        return sender

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

    @staticmethod
    def _extract_final_answer(content: str) -> str | None:
        """Extract the <ANSWER>: block from log content."""
        answer_start = content.find("<ANSWER>:")
        if answer_start < 0:
            if "FINAL_ANSWER" in content:
                return ""
            return None

        answer_end = content.find("</ANSWER>", answer_start)
        if answer_end < 0:
            answer_end = content.find("\u2500" * 16, answer_start)
            if answer_end < 0:
                answer_end = len(content)

        return content[answer_start:answer_end]
