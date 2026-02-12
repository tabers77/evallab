"""PPE benchmark metrics for evaluating reward functions.

All metrics are pure functions: (data, reward_fn) -> MetricResult.
Uses only stdlib — zero external dependencies.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

from agent_eval.core.score import ScoreVector

from .models import BenchmarkDataset, MetricResult, PreferencePair, BenchmarkSample


# ---------------------------------------------------------------------------
# Internal math helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _rank(values: list[float]) -> list[float]:
    """Fractional ranks with tie averaging (1-based)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1

    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (Pearson on ranks)."""
    return _pearson(_rank(x), _rank(y))


def _kendall_tau_b(x: list[float], y: list[float]) -> float:
    """Kendall tau-b with tie corrections."""
    n = len(x)
    if n < 2:
        return 0.0

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    ties_xy = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]

            if dx == 0 and dy == 0:
                ties_xy += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1

    n_pairs = n * (n - 1) / 2
    n0_x = n_pairs - ties_y - ties_xy
    n0_y = n_pairs - ties_x - ties_xy

    # ties_x counts pairs tied only in x, ties_y only in y
    denom_x = concordant + discordant + ties_x
    denom_y = concordant + discordant + ties_y

    if denom_x == 0 or denom_y == 0:
        return 0.0

    return (concordant - discordant) / math.sqrt(denom_x * denom_y)


# ---------------------------------------------------------------------------
# Helper to compute per-domain breakdowns
# ---------------------------------------------------------------------------

def _group_pairs_by_domain(
    pairs: list[PreferencePair],
) -> dict[str, list[PreferencePair]]:
    groups: dict[str, list[PreferencePair]] = defaultdict(list)
    for p in pairs:
        groups[p.domain].append(p)
    return dict(groups)


def _group_samples_by_domain(
    samples: list[BenchmarkSample],
) -> dict[str, list[BenchmarkSample]]:
    groups: dict[str, list[BenchmarkSample]] = defaultdict(list)
    for s in samples:
        groups[s.domain].append(s)
    return dict(groups)


# ---------------------------------------------------------------------------
# Metric: pairwise accuracy
# ---------------------------------------------------------------------------

def _pairwise_accuracy_raw(
    pairs: list[PreferencePair],
    reward_fn: Callable[[ScoreVector], float],
) -> float:
    if not pairs:
        return 0.0
    correct = 0
    for pair in pairs:
        r_pref = reward_fn(pair.preferred)
        r_rej = reward_fn(pair.rejected)
        if r_pref > r_rej:
            correct += 1
        elif r_pref == r_rej:
            correct += 0.5  # ties count as half-correct
    return correct / len(pairs)


def pairwise_accuracy(
    dataset: BenchmarkDataset,
    reward_fn: Callable[[ScoreVector], float],
) -> MetricResult:
    """Fraction of pairs where reward(preferred) > reward(rejected)."""
    overall = _pairwise_accuracy_raw(dataset.pairs, reward_fn)
    per_domain = {}
    for domain, group in _group_pairs_by_domain(dataset.pairs).items():
        per_domain[domain] = _pairwise_accuracy_raw(group, reward_fn)

    return MetricResult(
        metric_name="pairwise_accuracy",
        value=overall,
        per_domain=per_domain,
        n_samples=len(dataset.pairs),
    )


# ---------------------------------------------------------------------------
# Metric: best-of-K
# ---------------------------------------------------------------------------

def _best_of_k_raw(
    samples: list[BenchmarkSample],
    reward_fn: Callable[[ScoreVector], float],
) -> float:
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        rewards = [reward_fn(sv) for sv in sample.score_vectors]
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
        gt_best = max(sample.ground_truth_scores)
        if gt_best > 0:
            total += sample.ground_truth_scores[best_idx] / gt_best
        elif sample.ground_truth_scores[best_idx] == gt_best:
            total += 1.0
    return total / len(samples)


def best_of_k(
    dataset: BenchmarkDataset,
    reward_fn: Callable[[ScoreVector], float],
) -> MetricResult:
    """Expected ground-truth quality of the reward-selected candidate."""
    overall = _best_of_k_raw(dataset.samples, reward_fn)
    per_domain = {}
    for domain, group in _group_samples_by_domain(dataset.samples).items():
        per_domain[domain] = _best_of_k_raw(group, reward_fn)

    return MetricResult(
        metric_name="best_of_k",
        value=overall,
        per_domain=per_domain,
        n_samples=len(dataset.samples),
    )


# ---------------------------------------------------------------------------
# Metric: Spearman rank correlation
# ---------------------------------------------------------------------------

def _correlation_raw(
    samples: list[BenchmarkSample],
    reward_fn: Callable[[ScoreVector], float],
    corr_fn: Callable[[list[float], list[float]], float],
) -> float:
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        rewards = [reward_fn(sv) for sv in sample.score_vectors]
        total += corr_fn(rewards, sample.ground_truth_scores)
    return total / len(samples)


def spearman_correlation(
    dataset: BenchmarkDataset,
    reward_fn: Callable[[ScoreVector], float],
) -> MetricResult:
    """Average Spearman rank correlation across samples."""
    overall = _correlation_raw(dataset.samples, reward_fn, _spearman_rho)
    per_domain = {}
    for domain, group in _group_samples_by_domain(dataset.samples).items():
        per_domain[domain] = _correlation_raw(group, reward_fn, _spearman_rho)

    return MetricResult(
        metric_name="spearman_correlation",
        value=overall,
        per_domain=per_domain,
        n_samples=len(dataset.samples),
    )


# ---------------------------------------------------------------------------
# Metric: Kendall tau
# ---------------------------------------------------------------------------

def kendall_tau(
    dataset: BenchmarkDataset,
    reward_fn: Callable[[ScoreVector], float],
) -> MetricResult:
    """Average Kendall tau-b across samples."""
    overall = _correlation_raw(dataset.samples, reward_fn, _kendall_tau_b)
    per_domain = {}
    for domain, group in _group_samples_by_domain(dataset.samples).items():
        per_domain[domain] = _correlation_raw(group, reward_fn, _kendall_tau_b)

    return MetricResult(
        metric_name="kendall_tau",
        value=overall,
        per_domain=per_domain,
        n_samples=len(dataset.samples),
    )


# ---------------------------------------------------------------------------
# Metric: separability
# ---------------------------------------------------------------------------

def _separability_raw(
    pairs: list[PreferencePair],
    reward_fn: Callable[[ScoreVector], float],
    margin: float,
) -> float:
    if not pairs:
        return 0.0
    count = 0
    for pair in pairs:
        gap = reward_fn(pair.preferred) - reward_fn(pair.rejected)
        if gap > margin:
            count += 1
    return count / len(pairs)


def separability(
    dataset: BenchmarkDataset,
    reward_fn: Callable[[ScoreVector], float],
    margin: float = 0.1,
) -> MetricResult:
    """Fraction of pairs with reward gap exceeding the margin."""
    overall = _separability_raw(dataset.pairs, reward_fn, margin)
    per_domain = {}
    for domain, group in _group_pairs_by_domain(dataset.pairs).items():
        per_domain[domain] = _separability_raw(group, reward_fn, margin)

    return MetricResult(
        metric_name="separability",
        value=overall,
        per_domain=per_domain,
        n_samples=len(dataset.pairs),
        metadata={"margin": margin},
    )


# ---------------------------------------------------------------------------
# Metric: Brier score
# ---------------------------------------------------------------------------

def _brier_score_raw(
    pairs: list[PreferencePair],
    reward_fn: Callable[[ScoreVector], float],
) -> float:
    if not pairs:
        return 0.0
    total = 0.0
    for pair in pairs:
        gap = reward_fn(pair.preferred) - reward_fn(pair.rejected)
        prob = _sigmoid(gap)
        # Ground truth: preferred should win (label = 1)
        total += (prob - 1.0) ** 2
    return total / len(pairs)


def brier_score(
    dataset: BenchmarkDataset,
    reward_fn: Callable[[ScoreVector], float],
) -> MetricResult:
    """Sigmoid-calibrated mean squared error (lower is better)."""
    overall = _brier_score_raw(dataset.pairs, reward_fn)
    per_domain = {}
    for domain, group in _group_pairs_by_domain(dataset.pairs).items():
        per_domain[domain] = _brier_score_raw(group, reward_fn)

    return MetricResult(
        metric_name="brier_score",
        value=overall,
        per_domain=per_domain,
        n_samples=len(dataset.pairs),
    )


# ---------------------------------------------------------------------------
# Registry of all metrics
# ---------------------------------------------------------------------------

ALL_PAIR_METRICS = {
    "pairwise_accuracy": pairwise_accuracy,
    "separability": separability,
    "brier_score": brier_score,
}

ALL_SAMPLE_METRICS = {
    "best_of_k": best_of_k,
    "spearman_correlation": spearman_correlation,
    "kendall_tau": kendall_tau,
}

ALL_METRICS = {**ALL_PAIR_METRICS, **ALL_SAMPLE_METRICS}
