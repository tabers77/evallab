"""Bridge between agent_eval and DSPy MIPROv2.

Wraps agent_eval's evaluation pipeline as a DSPy metric function,
enabling MIPROv2 prompt optimization using eval scores as the
optimization signal.

DSPy expects metric functions with signature:
    (example, prediction, trace=None) -> float | bool

This module provides ``DSPyMetricBridge`` which:
1. Extracts the prompt and completion from DSPy's example/prediction
2. Builds a minimal Episode
3. Runs it through scorers
4. Returns a scalar score

Requires ``dspy`` as optional dependency.
"""

from __future__ import annotations

from typing import Any, Callable

try:
    import dspy  # noqa: F401

    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Issue, ScoreDimension, ScoreVector


class DSPyMetricBridge:
    """Wrap agent_eval evaluation as a DSPy metric function.

    Parameters
    ----------
    scorers
        List of Scorer instances to run on each episode.
    reward_fn
        A RewardFunction that converts ScoreVector to scalar.
    input_field
        The DSPy example field name for the input/prompt
        (default: ``"question"``).
    output_field
        The DSPy prediction field name for the output/completion
        (default: ``"answer"``).
    threshold
        When used as a boolean metric, scores above this threshold
        return True.  Set to None to always return the float score.
    episode_builder
        Optional custom function to build an Episode from
        ``(input_text, output_text)`` pairs.

    Example
    -------
    ::

        from agent_eval.rl.dspy_bridge import DSPyMetricBridge
        from agent_eval.rewards import WeightedSumReward
        from agent_eval.scorers.rules import IssueDetectorScorer

        metric = DSPyMetricBridge(
            scorers=[IssueDetectorScorer()],
            reward_fn=WeightedSumReward(),
        )

        # Use with DSPy MIPROv2
        teleprompter = dspy.MIPROv2(metric=metric.as_dspy_metric())
        optimized = teleprompter.compile(program, trainset=examples)
    """

    def __init__(
        self,
        scorers: list[Any],
        reward_fn: Any,
        input_field: str = "question",
        output_field: str = "answer",
        threshold: float | None = None,
        episode_builder: Callable[[str, str], Episode] | None = None,
    ) -> None:
        self.scorers = scorers
        self.reward_fn = reward_fn
        self.input_field = input_field
        self.output_field = output_field
        self.threshold = threshold
        self.episode_builder = episode_builder or _default_episode_builder

    def evaluate(self, input_text: str, output_text: str) -> float:
        """Compute a scalar score for an (input, output) pair."""
        episode = self.episode_builder(input_text, output_text)

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

    def __call__(
        self, example: Any, prediction: Any, trace: Any = None
    ) -> float | bool:
        """DSPy metric interface.

        Parameters
        ----------
        example
            A DSPy ``Example`` with the input field.
        prediction
            A DSPy ``Prediction`` with the output field.
        trace
            Optional DSPy trace (unused by this metric).

        Returns
        -------
        float or bool
            Float score if ``threshold`` is None, otherwise bool
            indicating whether the score exceeds the threshold.
        """
        input_text = self._extract_field(example, self.input_field)
        output_text = self._extract_field(prediction, self.output_field)

        score = self.evaluate(input_text, output_text)

        if self.threshold is not None:
            return score >= self.threshold
        return score

    def as_dspy_metric(self) -> Callable:
        """Return self as a callable for DSPy's metric parameter."""
        return self

    @staticmethod
    def _extract_field(obj: Any, field: str) -> str:
        """Extract a field from a DSPy Example/Prediction or plain dict."""
        if isinstance(obj, dict):
            return str(obj.get(field, ""))
        return str(getattr(obj, field, ""))


def _default_episode_builder(input_text: str, output_text: str) -> Episode:
    """Build a minimal Episode from an (input, output) pair."""
    steps = [
        Step(
            kind=StepKind.MESSAGE,
            agent_id="user",
            agent_name="user",
            content=input_text,
        ),
        Step(
            kind=StepKind.MESSAGE,
            agent_id="assistant",
            agent_name="assistant",
            content=output_text,
        ),
    ]
    return Episode(
        episode_id="dspy_episode",
        steps=steps,
        source_framework="dspy",
        final_answer=output_text,
    )
