"""Tests for rewards.composite — CompositeReward."""

import pytest

from agent_eval.core.score import Issue, ScoreDimension, ScoreVector, Severity
from agent_eval.rewards.aggregators import DeductionReward, WeightedSumReward
from agent_eval.rewards.composite import CompositeReward


def _make_sv(
    score: float = 0.8,
    n_warnings: int = 0,
    n_errors: int = 0,
    n_critical: int = 0,
) -> ScoreVector:
    dims = [ScoreDimension(name="overall_score", value=score, max_value=1.0)]
    issues = (
        [Issue(Severity.WARNING, "T", f"W{i}") for i in range(n_warnings)]
        + [Issue(Severity.ERROR, "T", f"E{i}") for i in range(n_errors)]
        + [Issue(Severity.CRITICAL, "T", f"C{i}") for i in range(n_critical)]
    )
    return ScoreVector(episode_id="test", dimensions=dims, issues=issues)


class TestCompositeReward:
    def test_single_component(self):
        ws = WeightedSumReward()
        comp = CompositeReward(components=[(ws, 1.0)])
        sv = _make_sv(0.8)
        assert abs(comp.compute(sv) - 0.8) < 1e-9

    def test_equal_weights(self):
        ws = WeightedSumReward()
        ded = DeductionReward()
        comp = CompositeReward(components=[(ws, 1.0), (ded, 1.0)])

        sv = _make_sv(0.75, n_warnings=2)
        ws_val = ws.compute(sv)
        ded_val = ded.compute(sv)
        expected = (ws_val + ded_val) / 2.0
        assert abs(comp.compute(sv) - expected) < 1e-9

    def test_unequal_weights(self):
        ws = WeightedSumReward()
        ded = DeductionReward()
        comp = CompositeReward(components=[(ws, 3.0), (ded, 1.0)])

        sv = _make_sv(0.9, n_errors=1)
        ws_val = ws.compute(sv)
        ded_val = ded.compute(sv)
        expected = (ws_val * 3.0 + ded_val * 1.0) / 4.0
        assert abs(comp.compute(sv) - expected) < 1e-9

    def test_no_normalize(self):
        ws = WeightedSumReward()
        comp = CompositeReward(components=[(ws, 2.0)], normalize=False)
        sv = _make_sv(0.6)
        expected = 0.6 * 2.0
        assert abs(comp.compute(sv) - expected) < 1e-9

    def test_empty_components_raises(self):
        with pytest.raises(ValueError, match="at least one component"):
            CompositeReward(components=[])

    def test_breakdown(self):
        ws = WeightedSumReward()
        ded = DeductionReward()
        comp = CompositeReward(components=[(ws, 1.0), (ded, 1.0)])

        sv = _make_sv(0.8, n_warnings=1)
        breakdown = comp.compute_breakdown(sv)
        assert "WeightedSumReward" in breakdown
        assert "DeductionReward" in breakdown
        assert abs(breakdown["WeightedSumReward"] - ws.compute(sv)) < 1e-9
        assert abs(breakdown["DeductionReward"] - ded.compute(sv)) < 1e-9

    def test_three_components(self):
        ws1 = WeightedSumReward()
        ws2 = WeightedSumReward(weights={"overall_score": 2.0})
        ded = DeductionReward()
        comp = CompositeReward(components=[(ws1, 1.0), (ws2, 1.0), (ded, 1.0)])
        sv = _make_sv(0.5, n_critical=1)
        result = comp.compute(sv)
        assert 0.0 <= result <= 1.0
