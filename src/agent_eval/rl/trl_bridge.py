"""Bridge between agent_eval and HuggingFace TRL.

Wraps EvalPipeline + RewardFunction into a callable compatible with
TRL's ``GRPOTrainer(reward_funcs=[...])``.

TRL expects reward functions with signature:
    (prompts: list[str], completions: list[str], **kwargs) -> list[float]

This module provides ``GRPORewardBridge`` which:
1. Converts each (prompt, completion) pair into a minimal Episode
2. Runs it through the eval pipeline's scorers
3. Returns scalar rewards via the configured RewardFunction

Requires ``trl`` as optional dependency (``pip install agent-eval[rl]``).
"""

from __future__ import annotations

from typing import Any, Callable

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Issue, ScoreDimension, ScoreVector

# Type alias for TRL-compatible reward function
TRLRewardFn = Callable[..., list[float]]

# Check for TRL availability
try:
    import trl  # noqa: F401

    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False


class GRPORewardBridge:
    """Wrap agent_eval evaluation as a TRL reward function.

    Parameters
    ----------
    scorers
        List of Scorer instances to run on each episode.
    reward_fn
        A RewardFunction that converts ScoreVector to scalar reward.
    episode_builder
        Optional custom function to build an Episode from
        ``(prompt, completion)`` pairs.  If not provided, a default
        builder creates a two-step Episode (user message + assistant
        message).

    Example
    -------
    ::

        from agent_eval.rl.trl_bridge import GRPORewardBridge
        from agent_eval.rewards import WeightedSumReward
        from agent_eval.scorers.rules import IssueDetectorScorer

        bridge = GRPORewardBridge(
            scorers=[IssueDetectorScorer()],
            reward_fn=WeightedSumReward(),
        )
        reward_fn = bridge.as_trl_reward_fn()

        # Use with TRL
        from trl import GRPOTrainer
        trainer = GRPOTrainer(..., reward_funcs=[reward_fn])
    """

    def __init__(
        self,
        scorers: list[Any],
        reward_fn: Any,
        episode_builder: Callable[[str, str], Episode] | None = None,
    ) -> None:
        self.scorers = scorers
        self.reward_fn = reward_fn
        self.episode_builder = episode_builder or _default_episode_builder

    def compute_reward(self, prompt: str, completion: str) -> float:
        """Compute a scalar reward for a single (prompt, completion) pair."""
        episode = self.episode_builder(prompt, completion)

        all_dimensions: list[ScoreDimension] = []
        all_issues: list[Issue] = []

        for scorer in self.scorers:
            all_dimensions.extend(scorer.score(episode))
            all_issues.extend(scorer.detect_issues(episode))

        score_vector = ScoreVector(
            episode_id=episode.episode_id,
            dimensions=all_dimensions,
            issues=all_issues,
        )

        return self.reward_fn.compute(score_vector)

    def compute_rewards(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        """Compute rewards for a batch — TRL-compatible signature."""
        if len(prompts) != len(completions):
            raise ValueError(
                f"prompts ({len(prompts)}) and completions ({len(completions)}) "
                "must have the same length"
            )
        return [self.compute_reward(p, c) for p, c in zip(prompts, completions)]

    def as_trl_reward_fn(self) -> TRLRewardFn:
        """Return a callable compatible with ``GRPOTrainer(reward_funcs=[...])``."""
        return self.compute_rewards


def _default_episode_builder(prompt: str, completion: str) -> Episode:
    """Build a minimal Episode from a (prompt, completion) pair."""
    steps = [
        Step(
            kind=StepKind.MESSAGE,
            agent_id="user",
            agent_name="user",
            content=prompt,
        ),
        Step(
            kind=StepKind.MESSAGE,
            agent_id="assistant",
            agent_name="assistant",
            content=completion,
        ),
    ]
    return Episode(
        episode_id="trl_episode",
        steps=steps,
        source_framework="trl",
        final_answer=completion,
    )
