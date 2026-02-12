"""Tests for experimental.ppe.synthetic — dataset generation."""

import pytest

from agent_eval.core.score import ScoreDimension, ScoreVector
from agent_eval.experimental.ppe.synthetic import (
    SyntheticDatasetBuilder,
    perturb_score_vector,
)


def _sv(eid: str, score: float) -> ScoreVector:
    return ScoreVector(
        episode_id=eid,
        dimensions=[ScoreDimension(name="q", value=score, max_value=1.0)],
    )


class TestPerturbScoreVector:
    def test_perturbed_has_different_id(self):
        base = _sv("ep1", 0.8)
        perturbed = perturb_score_vector(base, seed=42)
        assert perturbed.episode_id == "ep1_perturbed"

    def test_custom_episode_id(self):
        base = _sv("ep1", 0.8)
        perturbed = perturb_score_vector(base, episode_id="ep1_v2", seed=42)
        assert perturbed.episode_id == "ep1_v2"

    def test_values_clamped(self):
        base = _sv("ep1", 0.99)
        for seed in range(50):
            p = perturb_score_vector(base, noise_scale=0.5, seed=seed)
            for d in p.dimensions:
                assert 0.0 <= d.value <= d.max_value

    def test_deterministic_with_seed(self):
        base = _sv("ep1", 0.5)
        p1 = perturb_score_vector(base, seed=123)
        p2 = perturb_score_vector(base, seed=123)
        assert p1.dimensions[0].value == p2.dimensions[0].value

    def test_different_seeds_differ(self):
        base = _sv("ep1", 0.5)
        p1 = perturb_score_vector(base, seed=1)
        p2 = perturb_score_vector(base, seed=2)
        assert p1.dimensions[0].value != p2.dimensions[0].value

    def test_preserves_issues(self):
        from agent_eval.core.score import Issue, Severity
        base = ScoreVector(
            episode_id="ep1",
            dimensions=[ScoreDimension(name="q", value=0.5, max_value=1.0)],
            issues=[Issue(Severity.WARNING, "cat", "desc")],
        )
        perturbed = perturb_score_vector(base, seed=42)
        assert len(perturbed.issues) == 1


class TestSyntheticDatasetBuilder:
    def test_pairs_from_score_vectors(self):
        svs = [_sv(f"ep{i}", i * 0.1) for i in range(1, 6)]
        builder = SyntheticDatasetBuilder(seed=42)
        pairs = builder.pairs_from_score_vectors(svs)
        # 5 choose 2 = 10 pairs
        assert len(pairs) == 10
        for p in pairs:
            assert p.preferred.overall >= p.rejected.overall

    def test_pairs_with_quality_gap(self):
        svs = [_sv("a", 0.5), _sv("b", 0.51), _sv("c", 0.9)]
        builder = SyntheticDatasetBuilder(seed=42)
        pairs = builder.pairs_from_score_vectors(svs, min_quality_gap=0.1)
        # Only pairs with gap > 0.1 should remain
        for p in pairs:
            gap = abs(p.preferred.overall - p.rejected.overall)
            assert gap > 0.1

    def test_samples_from_score_vectors(self):
        svs = [_sv(f"ep{i}", i * 0.1) for i in range(1, 9)]
        builder = SyntheticDatasetBuilder(seed=42)
        samples = builder.samples_from_score_vectors(svs, k=4)
        assert len(samples) == 2  # 8 // 4
        for s in samples:
            assert len(s.score_vectors) == 4
            assert len(s.ground_truth_scores) == 4

    def test_samples_custom_n(self):
        svs = [_sv(f"ep{i}", i * 0.1) for i in range(1, 9)]
        builder = SyntheticDatasetBuilder(seed=42)
        samples = builder.samples_from_score_vectors(svs, k=2, n_samples=5)
        assert len(samples) == 5

    def test_samples_too_few_raises(self):
        svs = [_sv("a", 0.5)]
        builder = SyntheticDatasetBuilder(seed=42)
        with pytest.raises(ValueError, match="Need at least"):
            builder.samples_from_score_vectors(svs, k=4)

    def test_build_dataset(self):
        svs = [_sv(f"ep{i}", i * 0.1) for i in range(1, 9)]
        builder = SyntheticDatasetBuilder(seed=42)
        ds = builder.build_dataset(svs, name="synth_test", k=4)
        assert ds.name == "synth_test"
        assert len(ds.pairs) > 0
        assert len(ds.samples) > 0

    def test_custom_ground_truth(self):
        svs = [_sv(f"ep{i}", i * 0.1) for i in range(1, 5)]
        # Use a custom ground truth that inverts the score
        builder = SyntheticDatasetBuilder(
            ground_truth_fn=lambda sv: 1.0 - sv.overall,
            seed=42,
        )
        pairs = builder.pairs_from_score_vectors(svs)
        # The lowest-scoring sv should now be "preferred"
        for p in pairs:
            assert p.preferred.overall <= p.rejected.overall

    def test_reproducible_with_seed(self):
        svs = [_sv(f"ep{i}", i * 0.1) for i in range(1, 9)]
        b1 = SyntheticDatasetBuilder(seed=99)
        b2 = SyntheticDatasetBuilder(seed=99)
        s1 = b1.samples_from_score_vectors(svs, k=2, n_samples=3)
        s2 = b2.samples_from_score_vectors(svs, k=2, n_samples=3)
        for a, b in zip(s1, s2):
            ids_a = [sv.episode_id for sv in a.score_vectors]
            ids_b = [sv.episode_id for sv in b.score_vectors]
            assert ids_a == ids_b

    def test_domain_tag(self):
        svs = [_sv(f"ep{i}", i * 0.1) for i in range(1, 5)]
        builder = SyntheticDatasetBuilder(seed=42)
        pairs = builder.pairs_from_score_vectors(svs, domain="math")
        for p in pairs:
            assert p.domain == "math"
