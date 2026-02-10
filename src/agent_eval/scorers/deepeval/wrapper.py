"""Wrapper that adapts any DeepEval metric into the Scorer protocol.

Requires ``pip install agent-eval[deepeval]``. Fails gracefully at
import time if deepeval is not installed.

Usage::

    from agent_eval.scorers.deepeval.wrapper import DeepEvalScorer

    scorer = DeepEvalScorer(
        metric_name="AnswerRelevancyMetric",
        metric_kwargs={"threshold": 0.7, "model": "gpt-4o-mini"},
    )
    dims = scorer.score(episode)
"""

from __future__ import annotations

import logging
from typing import Any

from agent_eval.core.models import Episode, StepKind
from agent_eval.core.score import Issue, ScoreDimension, Severity

logger = logging.getLogger(__name__)

try:
    import deepeval  # noqa: F401

    _DEEPEVAL_AVAILABLE = True
except ImportError:
    _DEEPEVAL_AVAILABLE = False


class DeepEvalScorer:
    """Wrap a DeepEval metric as an agent_eval Scorer.

    Parameters
    ----------
    metric_name
        Name of the DeepEval metric class (e.g. ``"AnswerRelevancyMetric"``,
        ``"FaithfulnessMetric"``, ``"GEval"``).
    metric_kwargs
        Keyword arguments passed to the metric constructor.
    low_score_threshold
        Scores below this generate a WARNING issue.
    """

    def __init__(
        self,
        metric_name: str,
        metric_kwargs: dict[str, Any] | None = None,
        low_score_threshold: float = 0.5,
    ) -> None:
        if not _DEEPEVAL_AVAILABLE:
            raise ImportError(
                "deepeval is not installed. "
                "Install with: pip install agent-eval[deepeval]"
            )
        self.metric_name = metric_name
        self.metric_kwargs = metric_kwargs or {}
        self.low_score_threshold = low_score_threshold
        self._metric = self._create_metric()

    @property
    def name(self) -> str:
        return f"deepeval_{self.metric_name}"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        """Run the DeepEval metric and return a ScoreDimension."""
        from deepeval.test_case import LLMTestCase

        test_case = self._episode_to_test_case(episode)

        try:
            self._metric.measure(test_case)
            value = float(self._metric.score)
            value = max(0.0, min(1.0, value))
        except Exception as e:
            logger.warning("DeepEval metric %s failed: %s", self.metric_name, e)
            value = 0.0

        return [
            ScoreDimension(
                name=self.metric_name.lower(),
                value=value,
                max_value=1.0,
                source=self.name,
            )
        ]

    def detect_issues(self, episode: Episode) -> list[Issue]:
        """Flag low-scoring metrics as issues."""
        dims = self.score(episode)
        issues: list[Issue] = []
        for dim in dims:
            if dim.value < self.low_score_threshold:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        category="DeepEval",
                        description=(
                            f"Low {self.metric_name} score: "
                            f"{dim.value:.2f}/{dim.max_value:.2f}"
                        ),
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_metric(self):
        """Dynamically import and instantiate the DeepEval metric."""
        import deepeval.metrics as metrics_module

        metric_cls = getattr(metrics_module, self.metric_name, None)
        if metric_cls is None:
            raise ValueError(
                f"DeepEval metric '{self.metric_name}' not found. "
                f"Available: {[m for m in dir(metrics_module) if 'Metric' in m]}"
            )
        return metric_cls(**self.metric_kwargs)

    @staticmethod
    def _episode_to_test_case(episode: Episode):
        """Convert an Episode to a DeepEval LLMTestCase."""
        from deepeval.test_case import LLMTestCase

        # Build input from task description or first message
        input_text = episode.task_description or ""
        if not input_text:
            for step in episode.steps:
                if step.kind == StepKind.MESSAGE and step.content:
                    input_text = step.content
                    break

        # Build actual output from final answer
        actual_output = episode.final_answer or ""

        # Build retrieval context from tool results
        retrieval_context: list[str] = []
        for step in episode.steps:
            if step.kind == StepKind.TOOL_CALL and step.tool_result:
                retrieval_context.append(str(step.tool_result)[:500])

        return LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=retrieval_context if retrieval_context else None,
        )
