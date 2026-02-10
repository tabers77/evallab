"""Tests for scorers.deepeval.wrapper — DeepEvalScorer.

Since deepeval is an optional dependency, these tests verify the
import guard and basic structure without requiring deepeval installed.
"""

import pytest

from agent_eval.core.models import Episode, Step, StepKind


class TestDeepEvalImportGuard:
    def test_raises_import_error_when_not_installed(self):
        """DeepEvalScorer should raise ImportError if deepeval not installed."""
        try:
            import deepeval  # noqa: F401

            pytest.skip("deepeval is installed — skipping import guard test")
        except ImportError:
            pass

        from agent_eval.scorers.deepeval.wrapper import DeepEvalScorer

        with pytest.raises(ImportError, match="deepeval is not installed"):
            DeepEvalScorer(metric_name="AnswerRelevancyMetric")


class TestDeepEvalAvailability:
    def test_availability_flag(self):
        """Check that _DEEPEVAL_AVAILABLE reflects actual install state."""
        from agent_eval.scorers.deepeval.wrapper import _DEEPEVAL_AVAILABLE

        try:
            import deepeval  # noqa: F401

            assert _DEEPEVAL_AVAILABLE is True
        except ImportError:
            assert _DEEPEVAL_AVAILABLE is False
