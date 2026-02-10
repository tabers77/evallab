"""Composite reward: combine multiple RewardFunctions into one.

Generalises the prompt_tuner pattern of:
    reward = overall_score + lambda * semantic_similarity
into a flexible weighted combination of arbitrary reward components.
"""

from __future__ import annotations

from agent_eval.core.score import ScoreVector


class CompositeReward:
    """Weighted combination of multiple RewardFunction instances.

    Parameters
    ----------
    components
        Sequence of ``(reward_fn, weight)`` pairs.  Each ``reward_fn``
        must implement ``compute(score_vector) -> float``.
    normalize
        If *True* (default), the result is divided by the sum of
        weights so it stays in the same scale as the individual
        reward functions.  Set to *False* to get the raw weighted sum.
    """

    def __init__(
        self,
        components: list[tuple[object, float]],
        normalize: bool = True,
    ) -> None:
        if not components:
            raise ValueError("CompositeReward requires at least one component")
        self.components = components
        self.normalize = normalize

    def compute(self, score_vector: ScoreVector) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for reward_fn, weight in self.components:
            value = reward_fn.compute(score_vector)
            weighted_sum += value * weight
            total_weight += weight

        if self.normalize and total_weight > 0:
            return weighted_sum / total_weight
        return weighted_sum

    def compute_breakdown(self, score_vector: ScoreVector) -> dict[str, float]:
        """Return per-component reward values (unweighted) for diagnostics."""
        breakdown: dict[str, float] = {}
        for i, (reward_fn, _weight) in enumerate(self.components):
            name = getattr(reward_fn, "name", None) or type(reward_fn).__name__
            key = f"{name}_{i}" if name in breakdown else name
            breakdown[key] = reward_fn.compute(score_vector)
        return breakdown
