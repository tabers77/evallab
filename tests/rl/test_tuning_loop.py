"""Tests for rl.tuning_loop — TuningLoop."""

from agent_eval.core.score import ScoreDimension, ScoreVector
from agent_eval.rl.tuning_loop import TuningLoop


def _make_sv(score: float) -> ScoreVector:
    dims = [ScoreDimension(name="quality", value=score, max_value=1.0)]
    return ScoreVector(episode_id="test", dimensions=dims)


class FakeTestRunner:
    """Returns canned test results."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt):
        self.call_count += 1
        return [{"input": "q1", "output": f"answer_{self.call_count}"}]


class FakeEvaluator:
    """Returns improving scores each call."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.idx = 0

    def __call__(self, prompt, test_results):
        score = self.scores[min(self.idx, len(self.scores) - 1)]
        self.idx += 1
        return _make_sv(score)


class FakeEditor:
    """Appends iteration number to prompt."""

    def __call__(self, current_prompt, score_vector, test_results):
        new_prompt = f"{current_prompt}_improved"
        feedback = "Make it better"
        return new_prompt, feedback


class TestTuningLoop:
    def test_basic_run(self):
        runner = FakeTestRunner()
        evaluator = FakeEvaluator([0.6, 0.7, 0.8])
        editor = FakeEditor()
        reward_fn = lambda sv: sv.overall

        loop = TuningLoop(
            test_runner=runner,
            evaluator=evaluator,
            editor=editor,
            reward_fn=reward_fn,
            max_iterations=3,
        )
        result = loop.run("initial prompt")

        assert result.total_iterations == 3
        assert result.best_iteration == 3
        assert abs(result.best_reward - 0.8) < 1e-9
        assert "improved" in result.best_prompt

    def test_early_stopping(self):
        evaluator = FakeEvaluator([0.5, 0.95])
        loop = TuningLoop(
            test_runner=FakeTestRunner(),
            evaluator=evaluator,
            editor=FakeEditor(),
            reward_fn=lambda sv: sv.overall,
            max_iterations=10,
            target_reward=0.9,
        )
        result = loop.run("start")

        assert result.converged is True
        assert result.total_iterations == 2
        assert result.best_reward >= 0.9

    def test_lambda_similarity_bonus(self):
        evaluator = FakeEvaluator([0.5, 0.5])
        sim_fn = lambda a, b: 0.8  # constant similarity

        loop = TuningLoop(
            test_runner=FakeTestRunner(),
            evaluator=evaluator,
            editor=FakeEditor(),
            reward_fn=lambda sv: sv.overall,
            similarity_fn=sim_fn,
            max_iterations=2,
            lambda_param=0.5,
        )
        result = loop.run("start")

        # reward = 0.5 + 0.5 * 0.8 = 0.9
        assert abs(result.best_reward - 0.9) < 1e-9

    def test_single_iteration(self):
        evaluator = FakeEvaluator([0.75])
        loop = TuningLoop(
            test_runner=FakeTestRunner(),
            evaluator=evaluator,
            editor=FakeEditor(),
            reward_fn=lambda sv: sv.overall,
            max_iterations=1,
        )
        result = loop.run("start")

        assert result.total_iterations == 1
        assert result.best_iteration == 1

    def test_history_recorded(self):
        evaluator = FakeEvaluator([0.4, 0.6, 0.5])
        loop = TuningLoop(
            test_runner=FakeTestRunner(),
            evaluator=evaluator,
            editor=FakeEditor(),
            reward_fn=lambda sv: sv.overall,
            max_iterations=3,
        )
        result = loop.run("base")

        assert len(result.history) == 3
        assert result.history[0].iteration == 1
        assert result.history[1].iteration == 2
        assert result.history[2].iteration == 3
        # Best should be iteration 2 (score 0.6)
        assert result.best_iteration == 2

    def test_reward_trajectory(self):
        evaluator = FakeEvaluator([0.3, 0.7, 0.5])
        loop = TuningLoop(
            test_runner=FakeTestRunner(),
            evaluator=evaluator,
            editor=FakeEditor(),
            reward_fn=lambda sv: sv.overall,
            max_iterations=3,
        )
        result = loop.run("base")

        traj = result.reward_trajectory
        assert len(traj) == 3
        assert abs(traj[0] - 0.3) < 1e-9
        assert abs(traj[1] - 0.7) < 1e-9
        assert abs(traj[2] - 0.5) < 1e-9

    def test_no_target_no_convergence(self):
        evaluator = FakeEvaluator([0.99])
        loop = TuningLoop(
            test_runner=FakeTestRunner(),
            evaluator=evaluator,
            editor=FakeEditor(),
            reward_fn=lambda sv: sv.overall,
            max_iterations=1,
        )
        result = loop.run("start")
        # No target_reward set, so converged should be False
        assert result.converged is False
