"""Data models for RL-based prompt tuning.

Generalises the prompt_tunning/data_models.py structures:
- AggregateScores  -> ScoreVector  (already in core.score)
- IterationRecord  -> TuningIteration  (generic, uses ScoreVector)
- PromptTuningResult -> TuningResult   (generic)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_eval.core.score import ScoreVector


@dataclass
class TuningIteration:
    """Record for a single iteration of prompt tuning.

    Attributes
    ----------
    iteration
        1-based iteration number.
    prompt
        The prompt text used in this iteration.
    score_vector
        Full evaluation result from the pipeline.
    reward
        Scalar reward computed from the score vector.
    semantic_similarity
        Cosine similarity between the prompt and the feedback
        (measures how well the editor followed its own feedback).
    feedback_summary
        Human-readable summary of the editor's feedback.
    metadata
        Arbitrary extra data (e.g. editor model, timing info).
    """

    iteration: int
    prompt: str
    score_vector: ScoreVector
    reward: float
    semantic_similarity: float = 0.0
    feedback_summary: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "prompt": self.prompt,
            "score_vector": self.score_vector.to_dict(),
            "reward": round(self.reward, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "feedback_summary": self.feedback_summary,
            "metadata": self.metadata,
        }


@dataclass
class TuningResult:
    """Final result of a prompt tuning run.

    Attributes
    ----------
    best_prompt
        The prompt that achieved the highest reward.
    best_reward
        The reward value of the best prompt.
    best_iteration
        Which iteration produced the best prompt.
    history
        Full history of all iterations.
    converged
        Whether the tuning loop hit the target score.
    """

    best_prompt: str
    best_reward: float
    best_iteration: int
    history: list[TuningIteration] = field(default_factory=list)
    converged: bool = False

    @property
    def total_iterations(self) -> int:
        return len(self.history)

    @property
    def reward_trajectory(self) -> list[float]:
        """List of rewards across iterations for plotting."""
        return [it.reward for it in self.history]

    def to_dict(self) -> dict:
        return {
            "best_prompt": self.best_prompt,
            "best_reward": round(self.best_reward, 4),
            "best_iteration": self.best_iteration,
            "total_iterations": self.total_iterations,
            "converged": self.converged,
            "reward_trajectory": [round(r, 4) for r in self.reward_trajectory],
            "history": [it.to_dict() for it in self.history],
        }
