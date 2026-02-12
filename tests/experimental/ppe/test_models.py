"""Tests for experimental.ppe.models."""

import pytest

from agent_eval.core.score import ScoreDimension, ScoreVector
from agent_eval.experimental.ppe.models import (
    BenchmarkDataset,
    BenchmarkResult,
    BenchmarkSample,
    MetricResult,
    PreferencePair,
)


def _sv(eid: str, score: float = 0.8) -> ScoreVector:
    return ScoreVector(
        episode_id=eid,
        dimensions=[ScoreDimension(name="q", value=score, max_value=1.0)],
    )


class TestPreferencePair:
    def test_valid_pair(self):
        p = PreferencePair(preferred=_sv("a", 0.9), rejected=_sv("b", 0.3))
        assert p.domain == "default"

    def test_same_id_raises(self):
        with pytest.raises(ValueError, match="distinct episodes"):
            PreferencePair(preferred=_sv("a"), rejected=_sv("a"))

    def test_custom_domain(self):
        p = PreferencePair(preferred=_sv("a"), rejected=_sv("b"), domain="math")
        assert p.domain == "math"


class TestBenchmarkSample:
    def test_valid_sample(self):
        s = BenchmarkSample(
            score_vectors=[_sv("a"), _sv("b"), _sv("c")],
            ground_truth_scores=[0.9, 0.5, 0.2],
        )
        assert len(s.score_vectors) == 3

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match"):
            BenchmarkSample(
                score_vectors=[_sv("a"), _sv("b")],
                ground_truth_scores=[0.5],
            )

    def test_too_few_candidates_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            BenchmarkSample(
                score_vectors=[_sv("a")],
                ground_truth_scores=[0.5],
            )


class TestBenchmarkDataset:
    def test_with_pairs(self):
        pair = PreferencePair(preferred=_sv("a", 0.9), rejected=_sv("b", 0.3))
        ds = BenchmarkDataset(name="test", pairs=[pair])
        assert ds.domains == ["default"]

    def test_with_samples(self):
        sample = BenchmarkSample(
            score_vectors=[_sv("a"), _sv("b")],
            ground_truth_scores=[0.9, 0.3],
        )
        ds = BenchmarkDataset(name="test", samples=[sample])
        assert len(ds.samples) == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            BenchmarkDataset(name="empty")

    def test_multiple_domains(self):
        p1 = PreferencePair(preferred=_sv("a"), rejected=_sv("b"), domain="math")
        p2 = PreferencePair(preferred=_sv("c"), rejected=_sv("d"), domain="code")
        ds = BenchmarkDataset(name="multi", pairs=[p1, p2])
        assert ds.domains == ["code", "math"]

    def test_to_dict(self):
        pair = PreferencePair(preferred=_sv("a"), rejected=_sv("b"))
        ds = BenchmarkDataset(name="test", pairs=[pair])
        d = ds.to_dict()
        assert d["name"] == "test"
        assert d["n_pairs"] == 1
        assert d["n_samples"] == 0


class TestMetricResult:
    def test_to_dict(self):
        m = MetricResult(
            metric_name="pairwise_accuracy",
            value=0.85,
            per_domain={"math": 0.9, "code": 0.8},
            n_samples=100,
        )
        d = m.to_dict()
        assert d["metric_name"] == "pairwise_accuracy"
        assert d["value"] == 0.85
        assert d["n_samples"] == 100


class TestBenchmarkResult:
    def test_metric_by_name(self):
        m1 = MetricResult(metric_name="acc", value=0.9, n_samples=10)
        m2 = MetricResult(metric_name="brier", value=0.1, n_samples=10)
        br = BenchmarkResult(
            dataset_name="test", reward_fn_name="ws", metrics=[m1, m2]
        )
        assert br.metric_by_name("acc").value == 0.9
        assert br.metric_by_name("brier").value == 0.1
        assert br.metric_by_name("nonexistent") is None

    def test_to_dict(self):
        m = MetricResult(metric_name="acc", value=0.9, n_samples=10)
        br = BenchmarkResult(
            dataset_name="ds", reward_fn_name="fn", metrics=[m]
        )
        d = br.to_dict()
        assert d["dataset_name"] == "ds"
        assert d["reward_fn_name"] == "fn"
        assert len(d["metrics"]) == 1
