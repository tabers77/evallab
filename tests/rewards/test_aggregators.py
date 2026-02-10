"""Tests for rewards.aggregators — WeightedSumReward, DeductionReward."""

from agent_eval.core.score import Issue, ScoreDimension, ScoreVector, Severity
from agent_eval.rewards.aggregators import DeductionReward, WeightedSumReward


class TestWeightedSumReward:
    def test_equal_weights(self):
        sv = ScoreVector(
            episode_id="test",
            dimensions=[
                ScoreDimension(name="a", value=0.8),
                ScoreDimension(name="b", value=0.6),
            ],
        )
        reward = WeightedSumReward()
        assert reward.compute(sv) == 0.7

    def test_custom_weights(self):
        sv = ScoreVector(
            episode_id="test",
            dimensions=[
                ScoreDimension(name="accuracy", value=0.9),
                ScoreDimension(name="speed", value=0.3),
            ],
        )
        reward = WeightedSumReward(weights={"accuracy": 3.0, "speed": 1.0})
        # (0.9*3 + 0.3*1) / (3+1) = 3.0/4 = 0.75
        expected = (0.9 * 3 + 0.3 * 1) / 4
        assert abs(reward.compute(sv) - expected) < 1e-9

    def test_empty_dimensions(self):
        sv = ScoreVector(episode_id="test")
        reward = WeightedSumReward()
        assert reward.compute(sv) == 0.0


class TestDeductionReward:
    def test_no_issues_full_reward(self):
        sv = ScoreVector(episode_id="test")
        reward = DeductionReward()
        assert reward.compute(sv) == 1.0

    def test_critical_deduction(self):
        sv = ScoreVector(
            episode_id="test",
            issues=[Issue(Severity.CRITICAL, "Test", "C1")],
        )
        reward = DeductionReward()
        assert reward.compute(sv) == 0.75

    def test_mixed_deductions(self):
        sv = ScoreVector(
            episode_id="test",
            issues=[
                Issue(Severity.CRITICAL, "Test", "C1"),
                Issue(Severity.ERROR, "Test", "E1"),
                Issue(Severity.WARNING, "Test", "W1"),
            ],
        )
        reward = DeductionReward()
        # 1.0 - 0.25 - 0.10 - 0.05 = 0.60
        assert abs(reward.compute(sv) - 0.60) < 1e-9

    def test_clamped_to_zero(self):
        sv = ScoreVector(
            episode_id="test",
            issues=[Issue(Severity.CRITICAL, "Test", f"C{i}") for i in range(10)],
        )
        reward = DeductionReward()
        assert reward.compute(sv) == 0.0

    def test_custom_penalties(self):
        sv = ScoreVector(
            episode_id="test",
            issues=[Issue(Severity.CRITICAL, "Test", "C1")],
        )
        reward = DeductionReward(critical_penalty=0.5)
        assert reward.compute(sv) == 0.5
