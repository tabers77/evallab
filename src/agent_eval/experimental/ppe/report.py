"""Report formatting for PPE benchmark results."""

from __future__ import annotations

from .models import BenchmarkResult


def benchmark_to_dict(result: BenchmarkResult) -> dict:
    """Serialize a single benchmark result to a JSON-ready dict."""
    return result.to_dict()


def comparison_to_dict(results: list[BenchmarkResult]) -> dict:
    """Serialize multiple benchmark results for comparison."""
    return {
        "dataset_name": results[0].dataset_name if results else "",
        "reward_functions": [r.to_dict() for r in results],
    }


def benchmark_to_text(result: BenchmarkResult) -> str:
    """Format a single benchmark result as human-readable text."""
    lines = [
        f"PPE Benchmark: {result.dataset_name}",
        f"Reward Function: {result.reward_fn_name}",
        "-" * 50,
    ]

    for m in result.metrics:
        lines.append(f"  {m.metric_name:<25s} {m.value:>8.4f}  (n={m.n_samples})")
        if m.per_domain and len(m.per_domain) > 1:
            for domain, val in sorted(m.per_domain.items()):
                lines.append(f"    {domain:<23s} {val:>8.4f}")

    return "\n".join(lines)


def comparison_to_text(results: list[BenchmarkResult]) -> str:
    """Format multiple benchmark results as a side-by-side comparison table."""
    if not results:
        return "No results to compare."

    # Collect all metric names in order
    metric_names: list[str] = []
    for r in results:
        for m in r.metrics:
            if m.metric_name not in metric_names:
                metric_names.append(m.metric_name)

    fn_names = [r.reward_fn_name for r in results]

    # Header
    col_width = max(len(n) for n in fn_names) + 2
    col_width = max(col_width, 12)
    header = f"{'Metric':<25s}" + "".join(f"{n:>{col_width}s}" for n in fn_names)
    lines = [
        f"PPE Comparison: {results[0].dataset_name}",
        "=" * len(header),
        header,
        "-" * len(header),
    ]

    for metric_name in metric_names:
        row = f"{metric_name:<25s}"
        for r in results:
            m = r.metric_by_name(metric_name)
            if m is not None:
                row += f"{m.value:>{col_width}.4f}"
            else:
                row += f"{'N/A':>{col_width}s}"
        lines.append(row)

    return "\n".join(lines)
