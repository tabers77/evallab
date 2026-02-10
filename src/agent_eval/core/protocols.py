"""Protocol classes for structural typing (PEP 544).

Extensions never need to import from agent_eval to be compatible —
they just need to match the protocol shape.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_eval.core.models import Episode
from agent_eval.core.score import Issue, ScoreDimension, ScoreVector


@runtime_checkable
class TraceAdapter(Protocol):
    """Converts framework-specific traces into canonical Episodes."""

    @property
    def framework_name(self) -> str:
        ...

    def load_episode(self, source: str, **kwargs) -> Episode:
        ...

    def load_episodes(self, source: str, **kwargs) -> list[Episode]:
        ...


@runtime_checkable
class Scorer(Protocol):
    """Evaluates an Episode and returns score dimensions + issues."""

    @property
    def name(self) -> str:
        ...

    def score(self, episode: Episode) -> list[ScoreDimension]:
        ...

    def detect_issues(self, episode: Episode) -> list[Issue]:
        ...


@runtime_checkable
class RewardFunction(Protocol):
    """Converts a ScoreVector into a scalar reward for RL."""

    def compute(self, score_vector: ScoreVector) -> float:
        ...
