"""``agent-eval compare`` command.

Compares two evaluation runs side-by-side.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent_eval.adapters.autogen.adapter import AutoGenAdapter
from agent_eval.pipeline.runner import EvalPipeline
from agent_eval.pipeline.comparison import compare_results
from agent_eval.scorers.numeric.consistency import NumericConsistencyScorer
from agent_eval.scorers.rules.issue_detector import IssueDetectorScorer


def add_compare_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "compare",
        help="Compare two evaluation runs",
        description="Compare agent evaluation results from two log files.",
    )
    p.add_argument("path_a", help="Path to first log (baseline)")
    p.add_argument("path_b", help="Path to second log (experiment)")
    p.add_argument(
        "--label-a",
        default="Baseline",
        help="Label for first run (default: Baseline)",
    )
    p.add_argument(
        "--label-b",
        default="Experiment",
        help="Label for second run (default: Experiment)",
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


def run_compare(args: argparse.Namespace) -> None:
    """Execute the compare command."""
    adapter = AutoGenAdapter(agent_names=args.agents or [])
    scorers = [NumericConsistencyScorer(), IssueDetectorScorer()]
    pipeline = EvalPipeline(adapter=adapter, scorers=scorers)

    try:
        result_a = pipeline.evaluate_from_source(args.path_a)
        result_b = pipeline.evaluate_from_source(args.path_b)
    except Exception as e:
        print(f"Error loading logs: {e}", file=sys.stderr)
        sys.exit(1)

    comparison = compare_results(
        result_a,
        result_b,
        label_a=args.label_a,
        label_b=args.label_b,
    )

    if args.output_format == "json":
        print(json.dumps(comparison.to_dict(), indent=2, default=str))
    else:
        _print_text_comparison(comparison)


def _print_text_comparison(comp) -> None:
    """Print a human-readable comparison report."""
    lines = [
        "=" * 70,
        "COMPARISON REPORT",
        "=" * 70,
        "",
        f"  {comp.run_a_label:<30} vs  {comp.run_b_label}",
        "",
        f"  Overall Score:  {comp.run_a_score:>6.1f}       ->  {comp.run_b_score:>6.1f}  "
        f"(delta: {comp.score_delta:+.1f})",
        f"  Issues:         {comp.run_a_issue_count:>6d}       ->  {comp.run_b_issue_count:>6d}  "
        f"(delta: {comp.run_b_issue_count - comp.run_a_issue_count:+d})",
        "",
    ]

    if comp.improvements:
        lines.append("  Improvements:")
        for d in comp.improvements:
            lines.append(
                f"    + {d.name:<25} {d.run_a_value:.3f} -> {d.run_b_value:.3f} ({d.delta:+.4f})"
            )
        lines.append("")

    if comp.regressions:
        lines.append("  Regressions:")
        for d in comp.regressions:
            lines.append(
                f"    - {d.name:<25} {d.run_a_value:.3f} -> {d.run_b_value:.3f} ({d.delta:+.4f})"
            )
        lines.append("")

    verdict = (
        "IMPROVED"
        if comp.improved
        else "REGRESSED"
        if comp.score_delta < 0
        else "UNCHANGED"
    )
    lines.append(f"  Verdict: {verdict}")
    lines.append("=" * 70)

    print("\n".join(lines))
