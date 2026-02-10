"""Wrapper that adapts Ragas metrics into the Scorer protocol.

Requires ``pip install agent-eval[ragas]``. Fails gracefully at
import time if ragas is not installed.

Usage::

    from agent_eval.scorers.ragas.wrapper import RagasScorer

    scorer = RagasScorer(metric_name="faithfulness")
    dims = scorer.score(episode)
"""

from __future__ import annotations

import logging
from typing import Any

from agent_eval.core.models import Episode, StepKind
from agent_eval.core.score import Issue, ScoreDimension, Severity

logger = logging.getLogger(__name__)

try:
    import ragas  # noqa: F401

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False

# Known Ragas metric names → module paths
_RAGAS_METRICS = {
    "faithfulness": "ragas.metrics.faithfulness",
    "answer_relevancy": "ragas.metrics.answer_relevancy",
    "context_precision": "ragas.metrics.context_precision",
    "context_recall": "ragas.metrics.context_recall",
    "answer_similarity": "ragas.metrics.answer_similarity",
    "answer_correctness": "ragas.metrics.answer_correctness",
}


class RagasScorer:
    """Wrap a Ragas metric as an agent_eval Scorer.

    Parameters
    ----------
    metric_name
        Name of the Ragas metric (e.g. ``"faithfulness"``,
        ``"answer_relevancy"``).
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
        if not _RAGAS_AVAILABLE:
            raise ImportError(
                "ragas is not installed. " "Install with: pip install agent-eval[ragas]"
            )
        self.metric_name = metric_name
        self.metric_kwargs = metric_kwargs or {}
        self.low_score_threshold = low_score_threshold
        self._metric = self._create_metric()

    @property
    def name(self) -> str:
        return f"ragas_{self.metric_name}"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        """Run the Ragas metric and return a ScoreDimension."""
        from ragas import evaluate as ragas_evaluate
        from datasets import Dataset

        dataset = self._episode_to_dataset(episode)

        try:
            result = ragas_evaluate(dataset, metrics=[self._metric])
            # Ragas returns a dict with metric name -> score
            value = float(result[self.metric_name])
            value = max(0.0, min(1.0, value))
        except Exception as e:
            logger.warning("Ragas metric %s failed: %s", self.metric_name, e)
            value = 0.0

        return [
            ScoreDimension(
                name=self.metric_name,
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
                        category="Ragas",
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
        """Import and instantiate the Ragas metric."""
        import importlib

        if self.metric_name in _RAGAS_METRICS:
            module_path = _RAGAS_METRICS[self.metric_name]
            module = importlib.import_module(module_path)
            return getattr(module, self.metric_name)
        # Try direct import from ragas.metrics
        try:
            from ragas import metrics as ragas_metrics

            metric = getattr(ragas_metrics, self.metric_name, None)
            if metric is not None:
                return metric
        except AttributeError:
            pass
        raise ValueError(
            f"Ragas metric '{self.metric_name}' not found. "
            f"Known metrics: {list(_RAGAS_METRICS.keys())}"
        )

    @staticmethod
    def _episode_to_dataset(episode: Episode):
        """Convert an Episode to a Ragas-compatible HuggingFace Dataset."""
        from datasets import Dataset

        question = episode.task_description or ""
        if not question:
            for step in episode.steps:
                if step.kind == StepKind.MESSAGE and step.content:
                    question = step.content
                    break

        answer = episode.final_answer or ""

        contexts: list[str] = []
        for step in episode.steps:
            if step.kind == StepKind.TOOL_CALL and step.tool_result:
                contexts.append(str(step.tool_result)[:500])

        return Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
        )
