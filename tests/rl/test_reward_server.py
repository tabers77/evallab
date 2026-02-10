"""Tests for rl.reward_server — HTTP reward server.

Tests verify the server creation and import guards without requiring
FastAPI to be installed in all test environments.
"""

import pytest

from agent_eval.core.score import ScoreDimension, ScoreVector


class TestRewardServerImportGuard:
    def test_fastapi_availability_flag(self):
        from agent_eval.rl.reward_server import _FASTAPI_AVAILABLE

        # Just check the flag is a boolean — value depends on environment
        assert isinstance(_FASTAPI_AVAILABLE, bool)

    def test_create_app_raises_without_fastapi(self):
        from agent_eval.rl.reward_server import _FASTAPI_AVAILABLE

        if _FASTAPI_AVAILABLE:
            pytest.skip("fastapi is installed — skipping import guard test")

        from agent_eval.rl.reward_server import create_app

        with pytest.raises(ImportError, match="fastapi is not installed"):
            create_app(scorers=[], reward_fn=lambda sv: 0.0)


class FakeScorer:
    @property
    def name(self):
        return "fake"

    def score(self, episode):
        return [ScoreDimension(name="q", value=0.9, max_value=1.0)]

    def detect_issues(self, episode):
        return []


class FakeRewardFn:
    def compute(self, score_vector):
        return score_vector.overall


class TestRewardServerWithFastAPI:
    def test_create_app(self):
        """Test app creation when fastapi is available."""
        from agent_eval.rl.reward_server import _FASTAPI_AVAILABLE

        if not _FASTAPI_AVAILABLE:
            pytest.skip("fastapi not installed")

        from agent_eval.rl.reward_server import create_app

        app = create_app(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        assert app is not None
        assert app.title == "agent-eval Reward Server"

    def test_health_endpoint(self):
        """Test /health endpoint returns scorer names."""
        from agent_eval.rl.reward_server import _FASTAPI_AVAILABLE

        if not _FASTAPI_AVAILABLE:
            pytest.skip("fastapi not installed")

        from fastapi.testclient import TestClient
        from agent_eval.rl.reward_server import create_app

        app = create_app(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "fake" in data["scorers"]

    def test_reward_endpoint(self):
        """Test /reward endpoint returns rewards."""
        from agent_eval.rl.reward_server import _FASTAPI_AVAILABLE

        if not _FASTAPI_AVAILABLE:
            pytest.skip("fastapi not installed")

        from fastapi.testclient import TestClient
        from agent_eval.rl.reward_server import create_app

        app = create_app(
            scorers=[FakeScorer()],
            reward_fn=FakeRewardFn(),
        )
        client = TestClient(app)
        resp = client.post(
            "/reward",
            json={
                "prompts": ["What is 2+2?", "Hello"],
                "completions": ["4", "Hi there"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["rewards"]) == 2
        assert all(isinstance(r, float) for r in data["rewards"])
