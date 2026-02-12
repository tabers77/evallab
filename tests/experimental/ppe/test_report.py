"""Tests for experimental.ppe.report — text and dict formatting."""

from agent_eval.experimental.ppe.models import BenchmarkResult, MetricResult
from agent_eval.experimental.ppe.report import (
    benchmark_to_dict,
    benchmark_to_text,
    comparison_to_dict,
    comparison_to_text,
)


def _make_result(name: str = "WS", acc: float = 0.85, brier: float = 0.12) -> BenchmarkResult:
    return BenchmarkResult(
        dataset_name="test_ds",
        reward_fn_name=name,
        metrics=[
            MetricResult(
                metric_name="pairwise_accuracy",
                value=acc,
                per_domain={"math": 0.9, "code": 0.8},
                n_samples=100,
            ),
            MetricResult(
                metric_name="brier_score",
                value=brier,
                n_samples=100,
            ),
        ],
    )


class TestBenchmarkToDict:
    def test_structure(self):
        result = _make_result()
        d = benchmark_to_dict(result)
        assert d["dataset_name"] == "test_ds"
        assert d["reward_fn_name"] == "WS"
        assert len(d["metrics"]) == 2

    def test_values_rounded(self):
        result = _make_result(acc=0.85678)
        d = benchmark_to_dict(result)
        assert d["metrics"][0]["value"] == 0.8568


class TestComparisonToDict:
    def test_structure(self):
        r1 = _make_result("WS", 0.85, 0.12)
        r2 = _make_result("Ded", 0.70, 0.25)
        d = comparison_to_dict([r1, r2])
        assert d["dataset_name"] == "test_ds"
        assert len(d["reward_functions"]) == 2

    def test_empty(self):
        d = comparison_to_dict([])
        assert d["dataset_name"] == ""


class TestBenchmarkToText:
    def test_contains_key_info(self):
        result = _make_result()
        text = benchmark_to_text(result)
        assert "test_ds" in text
        assert "WS" in text
        assert "pairwise_accuracy" in text
        assert "brier_score" in text

    def test_per_domain_shown(self):
        result = _make_result()
        text = benchmark_to_text(result)
        assert "math" in text
        assert "code" in text


class TestComparisonToText:
    def test_contains_all_functions(self):
        r1 = _make_result("WS", 0.85, 0.12)
        r2 = _make_result("Ded", 0.70, 0.25)
        text = comparison_to_text([r1, r2])
        assert "WS" in text
        assert "Ded" in text
        assert "pairwise_accuracy" in text

    def test_empty_results(self):
        text = comparison_to_text([])
        assert "No results" in text
