"""``agent-eval evaluate`` command.

Evaluates one or more log files/directories and prints a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_eval.adapters.autogen.adapter import AutoGenAdapter
from agent_eval.pipeline.runner import EvalPipeline
from agent_eval.pipeline.batch import evaluate_paths, BatchResult
from agent_eval.reporting.text import format_report, format_comparison_report
from agent_eval.reporting.json_report import to_json, to_json_batch
from agent_eval.scorers.numeric.consistency import NumericConsistencyScorer
from agent_eval.scorers.rules.issue_detector import IssueDetectorScorer


def add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "evaluate",
        help="Evaluate agent log(s)",
        description="Score agent conversation logs and generate a report.",
    )
    p.add_argument(
        "paths",
        nargs="+",
        help="Path(s) to event.txt files or directories containing them",
    )
    p.add_argument(
        "--framework",
        choices=["autogen"],
        default="autogen",
        help="Agent framework (default: autogen)",
    )
    p.add_argument(
        "--agents",
        nargs="*",
        default=None,
        help="Known agent names for activity detection",
    )
    p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="output_format",
        help="Output format (default: text)",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write report to file instead of stdout",
    )
    p.add_argument(
        "--brief",
        action="store_true",
        help="Skip detailed metrics in text output",
    )
    p.add_argument(
        "--scorers",
        nargs="*",
        default=["numeric", "issues"],
        help="Scorers to run (default: numeric issues)",
    )


def run_evaluate(args: argparse.Namespace) -> None:
    """Execute the evaluate command."""
    adapter = _build_adapter(args)
    scorers = _build_scorers(args)
    pipeline = EvalPipeline(adapter=adapter, scorers=scorers)

    batch = evaluate_paths(pipeline, args.paths)

    if batch.count == 0:
        print("No logs found to evaluate.", file=sys.stderr)
        sys.exit(1)

    output = _format_output(batch, args)
    _write_output(output, args.output)


def _build_adapter(args: argparse.Namespace) -> AutoGenAdapter:
    return AutoGenAdapter(agent_names=args.agents or [])


def _build_scorers(args: argparse.Namespace) -> list:
    scorer_map = {
        "numeric": NumericConsistencyScorer,
        "issues": IssueDetectorScorer,
    }
    scorers = []
    for name in args.scorers:
        if name in scorer_map:
            scorers.append(scorer_map[name]())
        else:
            print(f"Warning: unknown scorer '{name}', skipping", file=sys.stderr)
    return scorers


def _format_output(batch: BatchResult, args: argparse.Namespace) -> str:
    if args.output_format == "json":
        return to_json_batch(batch.results)

    if batch.count == 1:
        return format_report(batch.results[0], verbose=not args.brief)
    return format_comparison_report(batch.results)


def _write_output(output: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
        print(f"Report saved to: {output_path}", file=sys.stderr)
    else:
        print(output)
