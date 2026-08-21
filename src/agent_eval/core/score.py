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
    # 0.3.0 — the judge is asked for a justification and always returned one;
    # it was parsed and thrown away, so a stored score could not be explained
    # or disputed after the fact (audit F8).
    justification: str | None = None
    # 0.3.0 — set when a scorer could NOT evaluate this dimension (API error,
    # unparseable response). Previously such a failure was recorded as
    # value=0.0, indistinguishable from a genuinely terrible answer, and it
    # dragged the aggregate down (audit F5). Abstained dimensions are excluded
    # from ``ScoreVector.overall``.
    abstained: bool = False

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
    def scored_dimensions(self) -> list[ScoreDimension]:
        """Dimensions that actually produced a score (excludes abstentions)."""
        return [d for d in self.dimensions if not d.abstained]

    @property
    def all_abstained(self) -> bool:
        """True when dimensions exist but none could be scored.

        A consumer must treat this as "not measured", NOT as a bad result —
        ``overall`` returns 0.0 in this case only because there is no honest
        number to return.
        """
        return bool(self.dimensions) and not self.scored_dimensions

    @property
    def overall(self) -> float:
        """Average of normalized values over the dimensions that scored.

        0.3.0 excludes abstained dimensions (audit F5). Previously a failed
        judge call was stored as 0.0 and averaged in, so transient API errors
        showed up as a genuine quality drop — measured at roughly -15 points,
        with 8.5% of records carrying the signature.

        Returns 0.0 if no dimensions are present, or if every dimension
        abstained; check :attr:`all_abstained` to tell those apart from a
        real zero.
        """
        scored = self.scored_dimensions
        if not scored:
            return 0.0
        return sum(d.normalized for d in scored) / len(scored)

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
                    "justification": d.justification,
                    "abstained": d.abstained,
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
