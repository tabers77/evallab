"""Tests for pipeline.runner — EvalPipeline."""

from agent_eval.adapters.autogen.adapter import AutoGenAdapter
from agent_eval.pipeline.runner import EvalPipeline, EvalResult
from agent_eval.scorers.numeric.consistency import NumericConsistencyScorer
from agent_eval.scorers.rules.issue_detector import IssueDetectorScorer


AGENT_NAMES = [
    "SalesNegotiator",
    "FinanceExpert",
    "CustomerResearcher",
    "DataVisualiser",
]


class TestEvalPipeline:
    def test_evaluate_episode(self, multi_agent_episode):
        pipeline = EvalPipeline(
            adapter=AutoGenAdapter(agent_names=AGENT_NAMES),
            scorers=[NumericConsistencyScorer(), IssueDetectorScorer()],
        )
        result = pipeline.evaluate(multi_agent_episode)

        assert isinstance(result, EvalResult)
        assert result.grade in ["A", "B", "C", "D", "F"]
        assert result.summary
        assert result.score_vector.episode_id == multi_agent_episode.episode_id

    def test_evaluate_from_source(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        pipeline = EvalPipeline(
            adapter=adapter,
            scorers=[NumericConsistencyScorer(), IssueDetectorScorer()],
        )
        result = pipeline.evaluate_from_source(str(sample_log_path))

        assert isinstance(result, EvalResult)
        assert result.grade in ["A", "B", "C", "D", "F"]
        assert result.score_vector.dimensions

    def test_evaluate_batch(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        pipeline = EvalPipeline(
            adapter=adapter,
            scorers=[IssueDetectorScorer()],
        )
        results = pipeline.evaluate_batch(str(sample_log_path.parent))

        assert len(results) == 1
        assert all(isinstance(r, EvalResult) for r in results)

    def test_score_vector_has_dimensions(self, multi_agent_episode):
        pipeline = EvalPipeline(
            adapter=AutoGenAdapter(),
            scorers=[NumericConsistencyScorer(), IssueDetectorScorer()],
        )
        result = pipeline.evaluate(multi_agent_episode)

        dim_names = {d.name for d in result.score_vector.dimensions}
        assert "numeric_accuracy" in dim_names
        assert "issue_free" in dim_names
        assert "overall_score" in dim_names

    def test_pipeline_with_no_scorers(self, multi_agent_episode):
        pipeline = EvalPipeline(
            adapter=AutoGenAdapter(),
            scorers=[],
        )
        result = pipeline.evaluate(multi_agent_episode)
        # Should still produce overall_score from RuleBasedScorer
        assert result.score_vector.dimension_by_name("overall_score") is not None
