"""Score models for multi-dimensional evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    """Issue severity levels."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Issue:
    """A detected issue in an agent trajectory."""

    severity: Severity
    category: str
    description: str
    line_number: int | None = None
    context: str | None = None


@dataclass
class ScoreDimension:
    """A single named dimension of evaluation.

    Scorers produce one or more ScoreDimensions per episode.
    """

    name: str
    value: float
    max_value: float = 1.0
    source: str = ""

    @property
    def normalized(self) -> float:
        """Value normalized to [0, 1]."""
        if self.max_value == 0:
            return 0.0
        return min(self.value / self.max_value, 1.0)


@dataclass
class ScoreVector:
    """Multi-dimensional evaluation result for an episode.

    Aggregates ScoreDimensions from multiple Scorers and
    Issues from issue detection, producing a single summary.
    """

    episode_id: str
    dimensions: list[ScoreDimension] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def overall(self) -> float:
        """Weighted average of normalized dimension values.

        Returns 0.0 if no dimensions are present.
        """
        if not self.dimensions:
            return 0.0
        total = sum(d.normalized for d in self.dimensions)
        return total / len(self.dimensions)

    def dimension_by_name(self, name: str) -> ScoreDimension | None:
        """Look up a dimension by name."""
        for d in self.dimensions:
            if d.name == name:
                return d
        return None

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "episode_id": self.episode_id,
            "overall": round(self.overall, 4),
            "dimensions": [
                {
                    "name": d.name,
                    "value": d.value,
                    "max_value": d.max_value,
                    "normalized": round(d.normalized, 4),
                    "source": d.source,
                }
                for d in self.dimensions
            ],
            "issues": [
                {
                    "severity": i.severity.value,
                    "category": i.category,
                    "description": i.description,
                    "line_number": i.line_number,
                    "context": i.context,
                }
                for i in self.issues
            ],
        }
