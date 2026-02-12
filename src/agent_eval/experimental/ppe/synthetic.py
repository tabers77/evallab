"""Synthetic dataset generation for PPE benchmarks."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

from agent_eval.core.score import ScoreDimension, ScoreVector

from .models import BenchmarkDataset, BenchmarkSample, PreferencePair


def perturb_score_vector(
    base: ScoreVector,
    noise_scale: float = 0.1,
    seed: int | None = None,
    episode_id: str | None = None,
) -> ScoreVector:
    """Create a noisy copy of a ScoreVector.

    Adds Gaussian noise to each dimension's value, clamped to
    ``[0, max_value]``.
    """
    rng = random.Random(seed)
    new_dims = []
    for d in base.dimensions:
        noise = rng.gauss(0, noise_scale * d.max_value)
        new_val = max(0.0, min(d.max_value, d.value + noise))
        new_dims.append(
            ScoreDimension(
                name=d.name,
                value=new_val,
                max_value=d.max_value,
                source=d.source,
            )
        )

    return ScoreVector(
        episode_id=episode_id or f"{base.episode_id}_perturbed",
        dimensions=new_dims,
        issues=list(base.issues),
    )


@dataclass
class SyntheticDatasetBuilder:
    """Builds PPE benchmark datasets from ScoreVectors.

    Parameters
    ----------
    ground_truth_fn
        Maps a ScoreVector to a scalar quality score.
        Defaults to ``ScoreVector.overall``.
    seed
        Random seed for reproducibility.
    """

    ground_truth_fn: Callable[[ScoreVector], float] | None = None
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.ground_truth_fn is None:
            self.ground_truth_fn = lambda sv: sv.overall

    def _gt(self, sv: ScoreVector) -> float:
        return self.ground_truth_fn(sv)

    def pairs_from_score_vectors(
        self,
        svs: list[ScoreVector],
        domain: str = "default",
        min_quality_gap: float = 0.0,
    ) -> list[PreferencePair]:
        """Generate all preference pairs above a quality gap threshold.

        Produces O(n^2) pairs from ``svs``, keeping only those where the
        ground-truth quality gap exceeds ``min_quality_gap``.
        """
        pairs = []
        for a, b in combinations(svs, 2):
            ga, gb = self._gt(a), self._gt(b)
            gap = abs(ga - gb)
            if gap <= min_quality_gap:
                continue
            if ga >= gb:
                pairs.append(PreferencePair(preferred=a, rejected=b, domain=domain))
            else:
                pairs.append(PreferencePair(preferred=b, rejected=a, domain=domain))
        return pairs

    def samples_from_score_vectors(
        self,
        svs: list[ScoreVector],
        k: int = 4,
        n_samples: int | None = None,
        domain: str = "default",
    ) -> list[BenchmarkSample]:
        """Generate benchmark samples by randomly grouping ScoreVectors.

        Parameters
        ----------
        svs
            Pool of ScoreVectors to sample from.
        k
            Number of candidates per sample.
        n_samples
            How many samples to generate.  Defaults to ``len(svs) // k``.
        domain
            Domain tag for generated samples.
        """
        if len(svs) < k:
            raise ValueError(
                f"Need at least {k} ScoreVectors to form samples of size {k}, "
                f"got {len(svs)}"
            )

        if n_samples is None:
            n_samples = len(svs) // k

        samples = []
        pool = list(svs)
        for _ in range(n_samples):
            chosen = self._rng.sample(pool, k)
            gt_scores = [self._gt(sv) for sv in chosen]
            samples.append(
                BenchmarkSample(
                    score_vectors=chosen,
                    ground_truth_scores=gt_scores,
                    domain=domain,
                )
            )
        return samples

    def build_dataset(
        self,
        svs: list[ScoreVector],
        name: str = "synthetic",
        k: int = 4,
        n_samples: int | None = None,
        min_quality_gap: float = 0.0,
        domain: str = "default",
    ) -> BenchmarkDataset:
        """Convenience method to build a complete dataset from ScoreVectors."""
        pairs = self.pairs_from_score_vectors(svs, domain, min_quality_gap)
        samples = self.samples_from_score_vectors(svs, k, n_samples, domain)
        return BenchmarkDataset(name=name, pairs=pairs, samples=samples)
