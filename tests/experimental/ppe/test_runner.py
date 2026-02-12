"""Tests for experimental.ppe.runner — BenchmarkRunner."""

import pytest

from agent_eval.core.score import ScoreDimension, ScoreVector, Issue, Severity
from agent_eval.rewards.aggregators import WeightedSumReward, DeductionReward
from agent_eval.experimental.ppe.models import (
    BenchmarkDataset,
    BenchmarkSample,
    PreferencePair,
)
from agent_eval.experimental.ppe.runner import BenchmarkRunner


def _sv(eid: str, score: float) -> ScoreVector:
    return ScoreVector(
        episode_id=eid,
        dimensions=[ScoreDimension(name="q", value=score, max_value=1.0)],
    )


def _full_dataset() -> BenchmarkDataset:
    pairs = [
        PreferencePair(preferred=_sv("a", 0.9), rejected=_sv("b", 0.1)),
        PreferencePair(preferred=_sv("c", 0.8), rejected=_sv("d", 0.2)),
    ]
    samples = [
        BenchmarkSample(
            score_vectors=[_sv("e", 0.9), _sv("f", 0.3), _sv("g", 0.1)],
            ground_truth_scores=[0.9, 0.3, 0.1],
        ),
    ]
    return BenchmarkDataset(name="test_ds", pairs=pairs, samples=samples)


class TestBenchmarkRunner:
    def test_run_all_metrics(self):
        ds = _full_dataset()
        runner = BenchmarkRunner(dataset=ds)
        result = runner.run(WeightedSumReward(), label="WS")
        assert result.reward_fn_name == "WS"
        assert result.dataset_name == "test_ds"
        # Should have all 6 metrics
        names = [m.metric_name for m in result.metrics]
        assert "pairwise_accuracy" in names
        assert "best_of_k" in names
        assert "spearman_correlation" in names
        assert "kendall_tau" in names
        assert "separability" in names
        assert "brier_score" in names

    def test_run_selected_metrics(self):
        ds = _full_dataset()
        runner = BenchmarkRunner(
            dataset=ds, metric_names=["pairwise_accuracy", "brier_score"]
        )
        result = runner.run(WeightedSumReward())
        names = [m.metric_name for m in result.metrics]
        assert names == ["pairwise_accuracy", "brier_score"]

    def test_run_pairs_only_dataset(self):
        pairs = [
            PreferencePair(preferred=_sv("a", 0.9), rejected=_sv("b", 0.1)),
        ]
        ds = BenchmarkDataset(name="pairs_only", pairs=pairs)
        runner = BenchmarkRunner(dataset=ds)
        result = runner.run(WeightedSumReward())
        names = [m.metric_name for m in result.metrics]
        assert "pairwise_accuracy" in names
        assert "best_of_k" not in names

    def test_run_samples_only_dataset(self):
        samples = [
            BenchmarkSample(
                score_vectors=[_sv("a", 0.9), _sv("b", 0.3)],
                ground_truth_scores=[0.9, 0.3],
            ),
        ]
        ds = BenchmarkDataset(name="samples_only", samples=samples)
        runner = BenchmarkRunner(dataset=ds)
        result = runner.run(WeightedSumReward())
        names = [m.metric_name for m in result.metrics]
        assert "best_of_k" in names
        assert "pairwise_accuracy" not in names

    def test_run_comparison(self):
        ds = _full_dataset()
        runner = BenchmarkRunner(dataset=ds)
        results = runner.run_comparison([
            (WeightedSumReward(), "WS"),
            (DeductionReward(), "Ded"),
        ])
        assert len(results) == 2
        assert results[0].reward_fn_name == "WS"
        assert results[1].reward_fn_name == "Ded"

    def test_auto_label(self):
        ds = _full_dataset()
        runner = BenchmarkRunner(dataset=ds, metric_names=["pairwise_accuracy"])
        result = runner.run(WeightedSumReward())
        assert result.reward_fn_name == "WeightedSumReward"

    def test_separability_margin_passed(self):
        pairs = [
            PreferencePair(preferred=_sv("a", 0.55), rejected=_sv("b", 0.50)),
        ]
        ds = BenchmarkDataset(name="test", pairs=pairs)
        # margin=0.1 -> gap=0.05 < margin -> sep=0
        runner = BenchmarkRunner(dataset=ds, separability_margin=0.1)
        result = runner.run(WeightedSumReward())
        sep = result.metric_by_name("separability")
        assert sep is not None
        assert sep.value == 0.0

        # margin=0.01 -> gap=0.05 > margin -> sep=1
        runner2 = BenchmarkRunner(dataset=ds, separability_margin=0.01)
        result2 = runner2.run(WeightedSumReward())
        sep2 = result2.metric_by_name("separability")
        assert sep2.value == 1.0
