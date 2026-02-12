"""Tests for experimental.ppe.metrics — validates all PPE math."""

import math

import pytest

from agent_eval.core.score import ScoreDimension, ScoreVector
from agent_eval.experimental.ppe.models import (
    BenchmarkDataset,
    BenchmarkSample,
    PreferencePair,
)
from agent_eval.experimental.ppe.metrics import (
    _kendall_tau_b,
    _rank,
    _sigmoid,
    _spearman_rho,
    best_of_k,
    brier_score,
    kendall_tau,
    pairwise_accuracy,
    separability,
    spearman_correlation,
)


def _sv(eid: str, score: float) -> ScoreVector:
    return ScoreVector(
        episode_id=eid,
        dimensions=[ScoreDimension(name="q", value=score, max_value=1.0)],
    )


def _pairs_dataset(pairs: list[PreferencePair]) -> BenchmarkDataset:
    """Build a dataset with only pairs (add a dummy sample to pass validation if needed)."""
    if not pairs:
        # Won't happen in our tests, but safeguard
        sample = BenchmarkSample(
            score_vectors=[_sv("x", 0.5), _sv("y", 0.5)],
            ground_truth_scores=[0.5, 0.5],
        )
        return BenchmarkDataset(name="test", samples=[sample])
    return BenchmarkDataset(name="test", pairs=pairs)


def _samples_dataset(samples: list[BenchmarkSample]) -> BenchmarkDataset:
    if not samples:
        pair = PreferencePair(preferred=_sv("x", 0.9), rejected=_sv("y", 0.1))
        return BenchmarkDataset(name="test", pairs=[pair])
    return BenchmarkDataset(name="test", samples=samples)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_zero(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-9

    def test_large_positive(self):
        assert abs(_sigmoid(100.0) - 1.0) < 1e-6

    def test_large_negative(self):
        assert abs(_sigmoid(-100.0)) < 1e-6

    def test_symmetry(self):
        assert abs(_sigmoid(2.0) + _sigmoid(-2.0) - 1.0) < 1e-9


class TestRank:
    def test_no_ties(self):
        assert _rank([10, 30, 20]) == [1.0, 3.0, 2.0]

    def test_ties(self):
        assert _rank([10, 10, 20]) == [1.5, 1.5, 3.0]

    def test_all_same(self):
        assert _rank([5, 5, 5]) == [2.0, 2.0, 2.0]

    def test_single(self):
        assert _rank([42]) == [1.0]


class TestSpearmanRho:
    def test_perfect_positive(self):
        assert abs(_spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9

    def test_perfect_negative(self):
        assert abs(_spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9

    def test_partial_correlation(self):
        # [1,2,3,4,5] vs [2,4,1,3,5]: d^2=[1,4,4,1,0], sum=10
        # rho = 1 - 6*10/(5*24) = 0.5
        rho = _spearman_rho([1, 2, 3, 4, 5], [2, 4, 1, 3, 5])
        assert abs(rho - 0.5) < 1e-9


class TestKendallTauB:
    def test_perfect_concordance(self):
        tau = _kendall_tau_b([1, 2, 3, 4], [10, 20, 30, 40])
        assert abs(tau - 1.0) < 1e-9

    def test_perfect_discordance(self):
        tau = _kendall_tau_b([1, 2, 3, 4], [40, 30, 20, 10])
        assert abs(tau + 1.0) < 1e-9

    def test_with_ties(self):
        tau = _kendall_tau_b([1, 1, 2, 3], [1, 2, 3, 4])
        assert -1.0 <= tau <= 1.0

    def test_single_pair(self):
        tau = _kendall_tau_b([1], [2])
        assert tau == 0.0  # n < 2


# ---------------------------------------------------------------------------
# PPE Metrics on datasets
# ---------------------------------------------------------------------------

class TestPairwiseAccuracy:
    def test_perfect_reward(self):
        """Reward fn that returns overall score gets all pairs right."""
        pairs = [
            PreferencePair(preferred=_sv("a", 0.9), rejected=_sv("b", 0.1)),
            PreferencePair(preferred=_sv("c", 0.8), rejected=_sv("d", 0.2)),
        ]
        ds = _pairs_dataset(pairs)
        result = pairwise_accuracy(ds, lambda sv: sv.overall)
        assert abs(result.value - 1.0) < 1e-9
        assert result.n_samples == 2

    def test_inverted_reward(self):
        """Reward fn that inverts scores gets all pairs wrong."""
        pairs = [
            PreferencePair(preferred=_sv("a", 0.9), rejected=_sv("b", 0.1)),
            PreferencePair(preferred=_sv("c", 0.8), rejected=_sv("d", 0.2)),
        ]
        ds = _pairs_dataset(pairs)
        result = pairwise_accuracy(ds, lambda sv: 1.0 - sv.overall)
        assert abs(result.value - 0.0) < 1e-9

    def test_tied_rewards(self):
        """Ties count as 0.5."""
        pairs = [
            PreferencePair(preferred=_sv("a", 0.5), rejected=_sv("b", 0.5)),
        ]
        ds = _pairs_dataset(pairs)
        result = pairwise_accuracy(ds, lambda sv: sv.overall)
        assert abs(result.value - 0.5) < 1e-9

    def test_per_domain(self):
        pairs = [
            PreferencePair(preferred=_sv("a", 0.9), rejected=_sv("b", 0.1), domain="math"),
            PreferencePair(preferred=_sv("c", 0.8), rejected=_sv("d", 0.2), domain="code"),
        ]
        ds = _pairs_dataset(pairs)
        result = pairwise_accuracy(ds, lambda sv: sv.overall)
        assert "math" in result.per_domain
        assert "code" in result.per_domain


class TestBestOfK:
    def test_perfect_selection(self):
        """Reward fn that matches ground truth picks the best candidate."""
        sample = BenchmarkSample(
            score_vectors=[_sv("a", 0.9), _sv("b", 0.3), _sv("c", 0.1)],
            ground_truth_scores=[0.9, 0.3, 0.1],
        )
        ds = _samples_dataset([sample])
        result = best_of_k(ds, lambda sv: sv.overall)
        assert abs(result.value - 1.0) < 1e-9

    def test_worst_selection(self):
        """Inverted reward selects the worst candidate."""
        sample = BenchmarkSample(
            score_vectors=[_sv("a", 0.9), _sv("b", 0.3), _sv("c", 0.1)],
            ground_truth_scores=[0.9, 0.3, 0.1],
        )
        ds = _samples_dataset([sample])
        result = best_of_k(ds, lambda sv: 1.0 - sv.overall)
        # Picks c (0.1) whose gt=0.1, normalized by best gt=0.9 -> ~0.111
        assert abs(result.value - 0.1 / 0.9) < 1e-9


class TestSpearmanCorrelation:
    def test_perfect_correlation(self):
        sample = BenchmarkSample(
            score_vectors=[_sv("a", 0.1), _sv("b", 0.5), _sv("c", 0.9)],
            ground_truth_scores=[0.1, 0.5, 0.9],
        )
        ds = _samples_dataset([sample])
        result = spearman_correlation(ds, lambda sv: sv.overall)
        assert abs(result.value - 1.0) < 1e-9

    def test_negative_correlation(self):
        sample = BenchmarkSample(
            score_vectors=[_sv("a", 0.1), _sv("b", 0.5), _sv("c", 0.9)],
            ground_truth_scores=[0.9, 0.5, 0.1],
        )
        ds = _samples_dataset([sample])
        result = spearman_correlation(ds, lambda sv: sv.overall)
        assert abs(result.value + 1.0) < 1e-9


class TestKendallTau:
    def test_perfect_concordance(self):
        sample = BenchmarkSample(
            score_vectors=[_sv("a", 0.1), _sv("b", 0.5), _sv("c", 0.9)],
            ground_truth_scores=[0.1, 0.5, 0.9],
        )
        ds = _samples_dataset([sample])
        result = kendall_tau(ds, lambda sv: sv.overall)
        assert abs(result.value - 1.0) < 1e-9


class TestSeparability:
    def test_all_separable(self):
        pairs = [
            PreferencePair(preferred=_sv("a", 0.9), rejected=_sv("b", 0.1)),
        ]
        ds = _pairs_dataset(pairs)
        result = separability(ds, lambda sv: sv.overall, margin=0.1)
        assert abs(result.value - 1.0) < 1e-9
        assert result.metadata["margin"] == 0.1

    def test_none_separable(self):
        pairs = [
            PreferencePair(preferred=_sv("a", 0.55), rejected=_sv("b", 0.50)),
        ]
        ds = _pairs_dataset(pairs)
        result = separability(ds, lambda sv: sv.overall, margin=0.1)
        assert abs(result.value - 0.0) < 1e-9

    def test_zero_margin(self):
        pairs = [
            PreferencePair(preferred=_sv("a", 0.51), rejected=_sv("b", 0.50)),
        ]
        ds = _pairs_dataset(pairs)
        result = separability(ds, lambda sv: sv.overall, margin=0.0)
        assert abs(result.value - 1.0) < 1e-9


class TestBrierScore:
    def test_good_separation(self):
        """Positive gap -> sigmoid > 0.5 -> brier < 0.25."""
        pairs = [
            PreferencePair(preferred=_sv("a", 0.99), rejected=_sv("b", 0.01)),
        ]
        ds = _pairs_dataset(pairs)
        result = brier_score(ds, lambda sv: sv.overall)
        # gap=0.98, sigmoid(0.98)≈0.727, brier≈0.0745
        assert result.value < 0.25

    def test_inverted_gives_high_brier(self):
        """Negative gap -> sigmoid < 0.5 -> brier > 0.25."""
        pairs = [
            PreferencePair(preferred=_sv("a", 0.99), rejected=_sv("b", 0.01)),
        ]
        ds = _pairs_dataset(pairs)
        result = brier_score(ds, lambda sv: 1.0 - sv.overall)
        # gap=-0.98, sigmoid(-0.98)≈0.273, brier≈0.529
        assert result.value > 0.25

    def test_tied_gives_quarter(self):
        """Zero gap -> sigmoid(0)=0.5 -> (0.5-1)^2 = 0.25."""
        pairs = [
            PreferencePair(preferred=_sv("a", 0.5), rejected=_sv("b", 0.5)),
        ]
        ds = _pairs_dataset(pairs)
        result = brier_score(ds, lambda sv: sv.overall)
        assert abs(result.value - 0.25) < 1e-9


class TestEmptyData:
    def test_pairwise_accuracy_no_pairs(self):
        sample = BenchmarkSample(
            score_vectors=[_sv("a", 0.5), _sv("b", 0.5)],
            ground_truth_scores=[0.5, 0.5],
        )
        ds = BenchmarkDataset(name="test", samples=[sample])
        result = pairwise_accuracy(ds, lambda sv: sv.overall)
        assert result.value == 0.0

    def test_best_of_k_no_samples(self):
        pair = PreferencePair(preferred=_sv("a", 0.9), rejected=_sv("b", 0.1))
        ds = BenchmarkDataset(name="test", pairs=[pair])
        result = best_of_k(ds, lambda sv: sv.overall)
        assert result.value == 0.0
