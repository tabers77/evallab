"""HTTP reward server for distributed RL (OpenRLHF pattern).

Exposes a ``/reward`` endpoint that accepts batches of
(prompt, completion) pairs and returns scalar rewards.

Requires ``fastapi`` and ``uvicorn`` — installed via
``pip install agent-eval[rl]``.

Usage::

    from agent_eval.rl.reward_server import create_app

    app = create_app(scorers=[...], reward_fn=WeightedSumReward())
    # uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from agent_eval.rl.trl_bridge import GRPORewardBridge


def _check_fastapi() -> None:
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "fastapi is not installed. " "Install with: pip install agent-eval[rl]"
        )


# Pydantic models defined conditionally to avoid import errors
if _FASTAPI_AVAILABLE:

    class RewardRequest(BaseModel):
        """Batch reward request.

        Accepts both plain strings and conversational message-dict
        lists (TRL v0.12+ format).
        """

        prompts: list
        completions: list

    class RewardResponse(BaseModel):
        """Batch reward response."""

        rewards: list[float]

    class HealthResponse(BaseModel):
        """Health check response."""

        status: str
        scorers: list[str]


def create_app(
    scorers: list[Any],
    reward_fn: Any,
    episode_builder: Any | None = None,
) -> Any:
    """Create a FastAPI app serving rewards.

    Parameters
    ----------
    scorers
        List of Scorer instances.
    reward_fn
        A RewardFunction to convert ScoreVector to scalar.
    episode_builder
        Optional custom Episode builder.

    Returns
    -------
    FastAPI
        A ready-to-run FastAPI application.
    """
    _check_fastapi()

    bridge = GRPORewardBridge(
        scorers=scorers,
        reward_fn=reward_fn,
        episode_builder=episode_builder,
    )

    app = FastAPI(
        title="agent-eval Reward Server",
        description="OpenRLHF-compatible reward API for RL training",
        version="0.1.0",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        scorer_names = [getattr(s, "name", type(s).__name__) for s in scorers]
        return HealthResponse(status="ok", scorers=scorer_names)

    @app.post("/reward", response_model=RewardResponse)
    def reward(request: RewardRequest) -> RewardResponse:
        rewards = bridge.compute_rewards(
            prompts=request.prompts,
            completions=request.completions,
        )
        return RewardResponse(rewards=rewards)

    return app


def run_server(
    scorers: list[Any],
    reward_fn: Any,
    host: str = "0.0.0.0",
    port: int = 8000,
    episode_builder: Any | None = None,
) -> None:
    """Create and run the reward server (blocking)."""
    _check_fastapi()
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is not installed. " "Install with: pip install agent-eval[rl]"
        )

    app = create_app(
        scorers=scorers,
        reward_fn=reward_fn,
        episode_builder=episode_builder,
    )
    uvicorn.run(app, host=host, port=port)
