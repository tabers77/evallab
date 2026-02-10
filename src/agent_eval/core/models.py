"""Canonical trajectory models for framework-agnostic evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class StepKind(str, Enum):
    """Classification of a single step in an agent trajectory."""

    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LLM_CALL = "llm_call"
    FACT_CHECK = "fact_check"
    CUSTOM = "custom"


@dataclass
class Step:
    """A single step in an agent trajectory.

    Framework adapters convert framework-specific events into Steps.
    """

    kind: StepKind
    agent_id: str
    agent_name: str
    content: str | None = None
    timestamp: datetime | None = None
    # Tool-related fields (when kind == TOOL_CALL / TOOL_RESULT)
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: Any = None
    tool_succeeded: bool | None = None
    # LLM-related fields (when kind == LLM_CALL)
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Episode:
    """A complete agent interaction trajectory.

    An Episode represents one full conversation or task execution,
    composed of ordered Steps from one or more agents.
    """

    episode_id: str
    steps: list[Step]
    source_framework: str
    task_description: str | None = None
    final_answer: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def agents(self) -> set[str]:
        """Unique agent names that participated in this episode."""
        return {s.agent_name for s in self.steps if s.agent_name}

    @property
    def duration_seconds(self) -> float | None:
        """Episode duration in seconds, or None if timestamps missing."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    def steps_by_kind(self, kind: StepKind) -> list[Step]:
        """Filter steps by kind."""
        return [s for s in self.steps if s.kind == kind]

    def steps_by_agent(self, agent_name: str) -> list[Step]:
        """Filter steps by agent name."""
        return [s for s in self.steps if s.agent_name == agent_name]
