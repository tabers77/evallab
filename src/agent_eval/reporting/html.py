"""HTML report output.

Generates self-contained HTML reports with inline CSS.
No external dependencies required.
"""

from __future__ import annotations

from agent_eval.core.models import StepKind
from agent_eval.core.score import Severity
from agent_eval.pipeline.runner import EvalResult


_SEVERITY_COLORS = {
    Severity.CRITICAL: "#dc3545",
    Severity.ERROR: "#fd7e14",
    Severity.WARNING: "#ffc107",
    Severity.INFO: "#17a2b8",
}

_GRADE_COLORS = {
    "A": "#28a745",
    "B": "#5cb85c",
    "C": "#ffc107",
    "D": "#fd7e14",
    "F": "#dc3545",
}


def format_html_report(result: EvalResult, verbose: bool = True) -> str:
    """Format an EvalResult as a self-contained HTML report."""
    ep = result.episode
    sv = result.score_vector

    overall_dim = sv.dimension_by_name("overall_score")
    overall_value = overall_dim.value if overall_dim else 0.0
    grade_color = _GRADE_COLORS.get(result.grade, "#6c757d")

    tool_steps = ep.steps_by_kind(StepKind.TOOL_CALL)
    unique_tools = {s.tool_name for s in tool_steps if s.tool_name}
    successful = sum(1 for s in tool_steps if s.tool_succeeded is not False)
    failed = sum(1 for s in tool_steps if s.tool_succeeded is False)
    llm_steps = ep.steps_by_kind(StepKind.LLM_CALL)

    # Issue counts
    critical = sum(1 for i in sv.issues if i.severity == Severity.CRITICAL)
    errors = sum(1 for i in sv.issues if i.severity == Severity.ERROR)
    warnings = sum(1 for i in sv.issues if i.severity == Severity.WARNING)
    infos = sum(1 for i in sv.issues if i.severity == Severity.INFO)

    parts: list[str] = []
    parts.append(_HTML_HEAD)
    parts.append("<body>")
    parts.append('<div class="container">')

    # Header
    parts.append(
        f"""
    <div class="header">
        <h1>Agent Evaluation Report</h1>
        <p class="source">Source: {_esc(ep.metadata.get('source_path', 'unknown'))}</p>
        <p class="source">Framework: {_esc(ep.source_framework)}</p>
    </div>
    """
    )

    # Score card
    parts.append(
        f"""
    <div class="score-card">
        <div class="score-value">{overall_value:.0f}/100</div>
        <div class="grade" style="background-color: {grade_color};">
            Grade: {_esc(result.grade)}
        </div>
    </div>
    """
    )

    # Summary
    parts.append(f'<div class="summary">{_esc(result.summary)}</div>')

    # Metrics table
    if verbose:
        parts.append(
            """
        <h2>Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
        """
        )
        parts.append(
            f"<tr><td>Agents Active</td><td>{', '.join(sorted(ep.agents))}</td></tr>"
        )
        parts.append(f"<tr><td>Tool Calls (Total)</td><td>{len(tool_steps)}</td></tr>")
        parts.append(f"<tr><td>Tool Calls (Successful)</td><td>{successful}</td></tr>")
        parts.append(f"<tr><td>Tool Calls (Failed)</td><td>{failed}</td></tr>")
        parts.append(f"<tr><td>Unique Tools</td><td>{len(unique_tools)}</td></tr>")
        parts.append(f"<tr><td>LLM Calls</td><td>{len(llm_steps)}</td></tr>")
        if ep.duration_seconds:
            parts.append(
                f"<tr><td>Duration</td><td>{ep.duration_seconds:.1f}s</td></tr>"
            )
        has_answer = "Yes" if ep.final_answer else "No"
        parts.append(f"<tr><td>Final Answer</td><td>{has_answer}</td></tr>")
        parts.append("</table>")

        # Score dimensions
        if sv.dimensions:
            parts.append(
                """
            <h2>Score Dimensions</h2>
            <table>
                <tr><th>Dimension</th><th>Value</th><th>Max</th><th>Normalized</th><th>Source</th></tr>
            """
            )
            for dim in sv.dimensions:
                bar_width = int(dim.normalized * 100)
                parts.append(
                    f"""
                <tr>
                    <td>{_esc(dim.name)}</td>
                    <td>{dim.value}</td>
                    <td>{dim.max_value}</td>
                    <td>
                        <div class="bar-container">
                            <div class="bar" style="width: {bar_width}%;">{dim.normalized:.2f}</div>
                        </div>
                    </td>
                    <td>{_esc(dim.source)}</td>
                </tr>
                """
                )
            parts.append("</table>")

    # Issues
    if sv.issues:
        parts.append(f"<h2>Issues ({len(sv.issues)})</h2>")
        if critical:
            parts.append(f'<span class="badge critical">{critical} Critical</span>')
        if errors:
            parts.append(f'<span class="badge error">{errors} Error</span>')
        if warnings:
            parts.append(f'<span class="badge warning">{warnings} Warning</span>')
        if infos:
            parts.append(f'<span class="badge info">{infos} Info</span>')

        by_category: dict[str, list] = {}
        for issue in sv.issues:
            by_category.setdefault(issue.category, []).append(issue)

        for category, cat_issues in sorted(by_category.items()):
            parts.append(f"<h3>{_esc(category)}</h3>")
            parts.append("<ul>")
            for issue in cat_issues:
                color = _SEVERITY_COLORS.get(issue.severity, "#6c757d")
                parts.append(
                    f'<li><span class="severity" style="color: {color};">'
                    f"[{issue.severity.value}]</span> {_esc(issue.description)}</li>"
                )
            parts.append("</ul>")
    else:
        parts.append('<div class="no-issues">No issues detected</div>')

    parts.append("</div>")  # container
    parts.append("</body></html>")
    return "\n".join(parts)


def format_html_batch(results: list[EvalResult]) -> str:
    """Format multiple EvalResults as an HTML comparison report."""
    if not results:
        return "<html><body><p>No results to display.</p></body></html>"

    scores = []
    for r in results:
        dim = r.score_vector.dimension_by_name("overall_score")
        scores.append(dim.value if dim else 0.0)

    avg = sum(scores) / len(scores)

    parts: list[str] = []
    parts.append(_HTML_HEAD)
    parts.append("<body>")
    parts.append('<div class="container">')
    parts.append("<h1>Batch Evaluation Report</h1>")
    parts.append(f"<p>Total runs: {len(results)} | Average score: {avg:.1f}/100</p>")

    parts.append(
        """
    <table>
        <tr><th>#</th><th>Source</th><th>Score</th><th>Grade</th><th>Issues</th></tr>
    """
    )

    sorted_results = sorted(zip(results, scores), key=lambda x: -x[1])
    for i, (result, score) in enumerate(sorted_results, 1):
        source = result.episode.metadata.get("source_path", "unknown")
        grade_color = _GRADE_COLORS.get(result.grade, "#6c757d")
        n_issues = len(result.score_vector.issues)
        parts.append(
            f"""
        <tr>
            <td>{i}</td>
            <td>{_esc(source)}</td>
            <td>{score:.0f}/100</td>
            <td><span class="grade-sm" style="background-color: {grade_color};">{result.grade}</span></td>
            <td>{n_issues}</td>
        </tr>
        """
        )

    parts.append("</table>")
    parts.append("</div></body></html>")
    return "\n".join(parts)


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Evaluation Report</title>
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }
    .container { max-width: 900px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .header h1 { margin-top: 0; color: #2c3e50; }
    .source { color: #666; margin: 2px 0; }
    .score-card { display: flex; align-items: center; gap: 20px; margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }
    .score-value { font-size: 2.5em; font-weight: bold; color: #2c3e50; }
    .grade { display: inline-block; padding: 8px 20px; color: #fff; border-radius: 6px; font-size: 1.2em; font-weight: bold; }
    .grade-sm { display: inline-block; padding: 2px 8px; color: #fff; border-radius: 4px; font-size: 0.9em; font-weight: bold; }
    .summary { white-space: pre-wrap; background: #f8f9fa; padding: 15px; border-radius: 6px; margin: 15px 0; font-family: monospace; font-size: 0.9em; }
    h2 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 8px; }
    h3 { color: #495057; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0; }
    th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }
    th { background: #f8f9fa; font-weight: 600; }
    tr:hover { background: #f8f9fa; }
    .bar-container { background: #e9ecef; border-radius: 4px; overflow: hidden; height: 22px; }
    .bar { background: #007bff; height: 100%; color: #fff; text-align: center; font-size: 0.8em; line-height: 22px; min-width: 35px; border-radius: 4px; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; color: #fff; font-size: 0.85em; margin: 2px; }
    .badge.critical { background: #dc3545; }
    .badge.error { background: #fd7e14; }
    .badge.warning { background: #ffc107; color: #333; }
    .badge.info { background: #17a2b8; }
    .severity { font-weight: bold; }
    .no-issues { color: #28a745; font-weight: bold; padding: 15px; background: #d4edda; border-radius: 6px; margin: 15px 0; }
    ul { padding-left: 20px; }
    li { margin: 5px 0; }
</style>
</head>"""
