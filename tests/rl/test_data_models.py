"""Tests for rl.data_models — TuningIteration, TuningResult."""

from agent_eval.core.score import ScoreDimension, ScoreVector
from agent_eval.rl.data_models import TuningIteration, TuningResult


def _make_sv(score: float = 0.8) -> ScoreVector:
    dims = [ScoreDimension(name="overall", value=score, max_value=1.0)]
    return ScoreVector(episode_id="test", dimensions=dims)


class TestTuningIteration:
    def test_basic_creation(self):
        sv = _make_sv(0.85)
        it = TuningIteration(
            iteration=1,
            prompt="test prompt",
            score_vector=sv,
            reward=0.85,
            semantic_similarity=0.72,
            feedback_summary="improve clarity",
        )
        assert it.iteration == 1
        assert it.reward == 0.85
        assert it.semantic_similarity == 0.72

    def test_to_dict(self):
        sv = _make_sv(0.9)
        it = TuningIteration(
            iteration=2,
            prompt="v2 prompt",
            score_vector=sv,
            reward=0.92,
        )
        d = it.to_dict()
        assert d["iteration"] == 2
        assert d["prompt"] == "v2 prompt"
        assert d["reward"] == 0.92
        assert "score_vector" in d
        assert "dimensions" in d["score_vector"]

    def test_default_values(self):
        sv = _make_sv()
        it = TuningIteration(iteration=1, prompt="p", score_vector=sv, reward=0.5)
        assert it.semantic_similarity == 0.0
        assert it.feedback_summary == ""
        assert it.metadata == {}


class TestTuningResult:
    def test_basic_creation(self):
        result = TuningResult(
            best_prompt="best",
            best_reward=0.95,
            best_iteration=3,
        )
        assert result.best_prompt == "best"
        assert result.total_iterations == 0
        assert result.converged is False

    def test_with_history(self):
        sv1 = _make_sv(0.7)
        sv2 = _make_sv(0.9)
        history = [
            TuningIteration(iteration=1, prompt="p1", score_vector=sv1, reward=0.7),
            TuningIteration(iteration=2, prompt="p2", score_vector=sv2, reward=0.9),
        ]
        result = TuningResult(
            best_prompt="p2",
            best_reward=0.9,
            best_iteration=2,
            history=history,
            converged=True,
        )
        assert result.total_iterations == 2
        assert result.reward_trajectory == [0.7, 0.9]
        assert result.converged is True

    def test_to_dict(self):
        sv = _make_sv(0.8)
        history = [
            TuningIteration(iteration=1, prompt="p", score_vector=sv, reward=0.8),
        ]
        result = TuningResult(
            best_prompt="p",
            best_reward=0.8,
            best_iteration=1,
            history=history,
        )
        d = result.to_dict()
        assert d["best_prompt"] == "p"
        assert d["best_reward"] == 0.8
        assert d["total_iterations"] == 1
        assert d["reward_trajectory"] == [0.8]
        assert len(d["history"]) == 1

    def test_empty_history(self):
        result = TuningResult(
            best_prompt="start",
            best_reward=0.0,
            best_iteration=0,
        )
        assert result.reward_trajectory == []
        assert result.total_iterations == 0
