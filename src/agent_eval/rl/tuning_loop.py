"""Generic prompt tuning loop.

Ports ``experiments/prompt_tunning/prompt_tuner.py`` into a
framework-agnostic loop with pluggable callbacks:

- **TestRunner**: run the agent with the current prompt
- **Evaluator**: produce EvalResult(s) from test outputs
- **Editor**: propose a new prompt given feedback
- **SimilarityFn**: measure alignment between feedback and new prompt

The loop tracks rewards via any ``RewardFunction`` and returns a
``TuningResult`` with the full iteration history.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, runtime_checkable

from agent_eval.core.score import ScoreVector
from agent_eval.rl.data_models import TuningIteration, TuningResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback protocols — users supply these
# ---------------------------------------------------------------------------


@runtime_checkable
class TestRunner(Protocol):
    """Run an agent with the given prompt and return raw results."""

    def __call__(self, prompt: str) -> list[dict[str, Any]]:
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Evaluate test outputs and return a ScoreVector."""

    def __call__(self, prompt: str, test_results: list[dict[str, Any]]) -> ScoreVector:
        ...


@runtime_checkable
class Editor(Protocol):
    """Propose a new prompt given current prompt, scores, and raw results.

    Returns ``(new_prompt, feedback_summary)``.
    """

    def __call__(
        self,
        current_prompt: str,
        score_vector: ScoreVector,
        test_results: list[dict[str, Any]],
    ) -> tuple[str, str]:
        ...


# Simpler callable types
SimilarityFn = Callable[[str, str], float]
RewardFn = Callable[[ScoreVector], float]


# ---------------------------------------------------------------------------
# Tuning loop
# ---------------------------------------------------------------------------


class TuningLoop:
    """Generic prompt tuning engine.

    Generalises PromptTuner: plug in callbacks for running tests,
    evaluating, editing prompts, and measuring similarity.

    Parameters
    ----------
    test_runner
        Runs the agent on tests with the current prompt.
    evaluator
        Produces a ScoreVector from test outputs.
    editor
        Proposes a new prompt given current scores and outputs.
    reward_fn
        Converts a ScoreVector to a scalar reward.
    similarity_fn
        Optional: measure alignment between new prompt and feedback.
        Defaults to a no-op returning 0.0.
    max_iterations
        Maximum number of tuning iterations.
    target_reward
        Stop early when reward reaches this threshold.
    lambda_param
        Weight for the semantic similarity bonus (added to the
        base reward: ``reward = base + lambda * similarity``).
    """

    def __init__(
        self,
        test_runner: TestRunner,
        evaluator: Evaluator,
        editor: Editor,
        reward_fn: RewardFn,
        similarity_fn: SimilarityFn | None = None,
        max_iterations: int = 5,
        target_reward: float | None = None,
        lambda_param: float = 0.0,
    ) -> None:
        self.test_runner = test_runner
        self.evaluator = evaluator
        self.editor = editor
        self.reward_fn = reward_fn
        self.similarity_fn = similarity_fn or (lambda _a, _b: 0.0)
        self.max_iterations = max_iterations
        self.target_reward = target_reward
        self.lambda_param = lambda_param

    def run(self, initial_prompt: str) -> TuningResult:
        """Execute the tuning loop.

        Parameters
        ----------
        initial_prompt
            Starting prompt text.

        Returns
        -------
        TuningResult
            Contains the best prompt, reward trajectory, and full history.
        """
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_reward = float("-inf")
        best_iteration = 0
        history: list[TuningIteration] = []
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            logger.info("=== Iteration %d / %d ===", iteration, self.max_iterations)

            # 1) Run agent on tests
            logger.info("Running tests...")
            test_results = self.test_runner(current_prompt)

            # 2) Evaluate outputs -> ScoreVector
            logger.info("Evaluating results...")
            score_vector = self.evaluator(current_prompt, test_results)

            # 3) Compute base reward
            base_reward = self.reward_fn(score_vector)

            # 4) Edit prompt
            logger.info("Editing prompt...")
            new_prompt, feedback_summary = self.editor(
                current_prompt, score_vector, test_results
            )

            # 5) Semantic similarity bonus
            similarity = self.similarity_fn(new_prompt, feedback_summary)
            reward = base_reward + self.lambda_param * similarity

            logger.info(
                "Reward: %.4f (base=%.4f, sim=%.4f, lambda=%.4f)",
                reward,
                base_reward,
                similarity,
                self.lambda_param,
            )

            # 6) Record iteration
            history.append(
                TuningIteration(
                    iteration=iteration,
                    prompt=new_prompt,
                    score_vector=score_vector,
                    reward=reward,
                    semantic_similarity=similarity,
                    feedback_summary=feedback_summary,
                )
            )

            # 7) Track best
            if reward > best_reward:
                logger.info(
                    "New best at iteration %d (%.4f > %.4f)",
                    iteration,
                    reward,
                    best_reward,
                )
                best_reward = reward
                best_prompt = new_prompt
                best_iteration = iteration

            # 8) Early stopping
            if self.target_reward is not None and reward >= self.target_reward:
                logger.info(
                    "Target reward reached (%.4f >= %.4f); stopping early.",
                    reward,
                    self.target_reward,
                )
                converged = True
                break

            current_prompt = new_prompt

        return TuningResult(
            best_prompt=best_prompt,
            best_reward=best_reward,
            best_iteration=best_iteration,
            history=history,
            converged=converged,
        )
