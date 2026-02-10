"""CLI entry point for agent-eval.

Uses argparse (zero dependencies) so the CLI works without installing
optional packages. Subcommands:
  - evaluate: Score a single log or directory
  - compare:  Compare two log evaluations
"""

from __future__ import annotations

import argparse
import sys

from agent_eval.cli.commands.evaluate import add_evaluate_parser, run_evaluate
from agent_eval.cli.commands.compare import add_compare_parser, run_compare
from agent_eval.cli.commands.serve import add_serve_parser, run_serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Framework-agnostic LLM/Agent evaluator",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    add_evaluate_parser(sub)
    add_compare_parser(sub)
    add_serve_parser(sub)

    return parser


def app(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    app()
