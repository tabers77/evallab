"""Reward aggregation functions for RL integration.

Convert multi-dimensional ScoreVectors into scalar rewards
suitable for RL training loops.
"""

from __future__ import annotations

from agent_eval.core.score import ScoreVector, Severity


class WeightedSumReward:
    """Weighted sum of score dimensions.

    Parameters
    ----------
    weights
        Mapping from dimension name to weight.  Dimensions not in the
        map get ``default_weight``.
    default_weight
        Weight for unspecified dimensions.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        default_weight: float = 1.0,
    ) -> None:
        self.weights = weights or {}
        self.default_weight = default_weight

    def compute(self, score_vector: ScoreVector) -> float:
        if not score_vector.dimensions:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for dim in score_vector.dimensions:
            w = self.weights.get(dim.name, self.default_weight)
            weighted_sum += dim.normalized * w
            total_weight += w

        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight


class DeductionReward:
    """Deduction-based reward matching the original log evaluator scoring.

    Base reward of 1.0, with deductions per issue severity,
    producing a reward in [0, 1].

    Parameters
    ----------
    critical_penalty
        Deduction per CRITICAL issue (default 0.25).
    error_penalty
        Deduction per ERROR issue (default 0.10).
    warning_penalty
        Deduction per WARNING issue (default 0.05).
    """

    def __init__(
        self,
        critical_penalty: float = 0.25,
        error_penalty: float = 0.10,
        warning_penalty: float = 0.05,
    ) -> None:
        self.critical_penalty = critical_penalty
        self.error_penalty = error_penalty
        self.warning_penalty = warning_penalty

    def compute(self, score_vector: ScoreVector) -> float:
        reward = 1.0
        for issue in score_vector.issues:
            if issue.severity == Severity.CRITICAL:
                reward -= self.critical_penalty
            elif issue.severity == Severity.ERROR:
                reward -= self.error_penalty
            elif issue.severity == Severity.WARNING:
                reward -= self.warning_penalty
        return max(0.0, min(1.0, reward))
