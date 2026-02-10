"""CLI command: agent-eval serve — launch the reward server."""

from __future__ import annotations

import argparse


def add_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "serve",
        help="Start the HTTP reward server for RL training",
    )
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    p.add_argument(
        "--scorers",
        nargs="+",
        default=["issue_detector"],
        help="Scorer names from registry (default: issue_detector)",
    )


def run_serve(args: argparse.Namespace) -> None:
    """Launch the reward server."""
    from agent_eval.scorers.registry import default_registry
    from agent_eval.rewards.aggregators import WeightedSumReward
    from agent_eval.rl.reward_server import run_server

    scorers = [default_registry.get(name) for name in args.scorers]
    reward_fn = WeightedSumReward()

    print(f"Starting reward server on {args.host}:{args.port}")
    print(f"Scorers: {[s.name for s in scorers]}")
    run_server(
        scorers=scorers,
        reward_fn=reward_fn,
        host=args.host,
        port=args.port,
    )
