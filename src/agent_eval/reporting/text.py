"""Terminal text report formatter."""

from __future__ import annotations

from pathlib import Path

from agent_eval.core.models import StepKind
from agent_eval.core.score import Severity
from agent_eval.pipeline.runner import EvalResult


def format_report(result: EvalResult, verbose: bool = True) -> str:
    """Format an EvalResult as a readable text report."""
    lines: list[str] = []
    ep = result.episode
    sv = result.score_vector

    overall_dim = sv.dimension_by_name("overall_score")
    overall_value = overall_dim.value if overall_dim else 0.0

    lines.append("=" * 80)
    lines.append("AGENT EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append(f"\nSource: {ep.metadata.get('source_path', 'unknown')}")
    lines.append(f"Framework: {ep.source_framework}")
    lines.append(
        f"\n{'OVERALL SCORE:':<20} {overall_value}/100 (Grade: {result.grade})"
    )
    lines.append("=" * 80)

    lines.append(f"\n{result.summary}")

    if verbose:
        lines.append("\n" + "-" * 80)
        lines.append("DETAILED METRICS")
        lines.append("-" * 80)

        lines.append(f"{'Agents Active:':<30} {', '.join(sorted(ep.agents))}")

        tool_steps = ep.steps_by_kind(StepKind.TOOL_CALL)
        unique_tools = {s.tool_name for s in tool_steps if s.tool_name}
        successful = sum(1 for s in tool_steps if s.tool_succeeded is not False)
        failed = sum(1 for s in tool_steps if s.tool_succeeded is False)
        llm_steps = ep.steps_by_kind(StepKind.LLM_CALL)
        msg_steps = ep.steps_by_kind(StepKind.MESSAGE)
        non_empty = sum(1 for s in msg_steps if s.content and s.content.strip())

        lines.append(f"{'Total Turns:':<30} {non_empty}")
        lines.append(f"{'Tool Calls (Total):':<30} {len(tool_steps)}")
        lines.append(f"{'Tool Calls (Successful):':<30} {successful}")
        lines.append(f"{'Tool Calls (Failed):':<30} {failed}")
        lines.append(f"{'Unique Tools Used:':<30} {len(unique_tools)}")
        if unique_tools:
            lines.append(f"  Tools: {', '.join(sorted(unique_tools))}")
        lines.append(f"{'LLM Calls:':<30} {len(llm_steps)}")

        if ep.duration_seconds:
            lines.append(
                f"{'Execution Time:':<30} {ep.duration_seconds:.1f}s "
                f"({ep.duration_seconds / 60:.1f} min)"
            )

        lines.append(
            f"{'Final Answer Present:':<30} " f"{'Yes' if ep.final_answer else 'No'}"
        )
        if ep.final_answer:
            lines.append(f"{'Final Answer Length:':<30} {len(ep.final_answer)} chars")

        # Score dimensions
        lines.append("\n" + "-" * 80)
        lines.append("SCORE DIMENSIONS")
        lines.append("-" * 80)
        for dim in sv.dimensions:
            lines.append(
                f"  {dim.name:<30} {dim.value}/{dim.max_value} "
                f"(normalized: {dim.normalized:.3f}) [{dim.source}]"
            )

    # Issues
    if sv.issues:
        lines.append("\n" + "-" * 80)
        lines.append(f"ISSUES DETECTED ({len(sv.issues)})")
        lines.append("-" * 80)

        by_category: dict[str, list] = {}
        for issue in sv.issues:
            by_category.setdefault(issue.category, []).append(issue)

        severity_symbol = {
            Severity.CRITICAL: "[CRITICAL]",
            Severity.ERROR: "[ERROR]",
            Severity.WARNING: "[WARNING]",
            Severity.INFO: "[INFO]",
        }

        for category, cat_issues in sorted(by_category.items()):
            lines.append(f"\n{category}:")
            for issue in cat_issues:
                sym = severity_symbol.get(issue.severity, "[?]")
                line_info = f" (Line {issue.line_number})" if issue.line_number else ""
                lines.append(f"  {sym} {issue.description}{line_info}")
                if verbose and issue.context:
                    preview = (
                        issue.context[:150] + "..."
                        if len(issue.context) > 150
                        else issue.context
                    )
                    lines.append(f"     Context: {preview}")
    else:
        lines.append("\n" + "-" * 80)
        lines.append("NO ISSUES DETECTED")
        lines.append("-" * 80)

    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def format_comparison_report(results: list[EvalResult]) -> str:
    """Format a comparison report for multiple evaluation results."""
    if not results:
        return "No results to compare."

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("MULTI-RUN COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append(f"\nTotal Runs Evaluated: {len(results)}")

    scores = []
    for r in results:
        dim = r.score_vector.dimension_by_name("overall_score")
        scores.append(dim.value if dim else 0.0)

    avg = sum(scores) / len(scores)
    lines.append(f"\n{'Average Score:':<30} {avg:.1f}/100")
    lines.append(f"{'Best Score:':<30} {max(scores):.1f}")
    lines.append(f"{'Worst Score:':<30} {min(scores):.1f}")

    grades: dict[str, int] = {}
    for r in results:
        grades[r.grade] = grades.get(r.grade, 0) + 1

    lines.append(f"\nGrade Distribution:")
    for grade in ["A", "B", "C", "D", "F"]:
        count = grades.get(grade, 0)
        if count > 0:
            bar = "#" * count
            lines.append(f"  {grade}: {bar} ({count})")

    lines.append("\n" + "-" * 80)
    lines.append("INDIVIDUAL RESULTS")
    lines.append("-" * 80)

    sorted_results = sorted(zip(results, scores), key=lambda x: -x[1])
    for i, (result, score) in enumerate(sorted_results, 1):
        source = result.episode.metadata.get("source_path", "unknown")
        source_name = Path(source).parent.name if source != "unknown" else "unknown"
        n_issues = len(result.score_vector.issues)
        lines.append(f"\n{i}. {source_name}")
        lines.append(f"   Score: {score}/100 (Grade: {result.grade})")
        lines.append(f"   Issues: {n_issues} total")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
