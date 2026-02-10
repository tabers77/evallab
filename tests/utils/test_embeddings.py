"""Tests for utils.embeddings — cosine similarity and helpers."""

import pytest


class TestCosineSimImportGuard:
    def test_numpy_availability(self):
        from agent_eval.utils.embeddings import _NUMPY_AVAILABLE

        # numpy is typically installed; just check the flag exists
        assert isinstance(_NUMPY_AVAILABLE, bool)


class TestCosineSimilarity:
    def _skip_if_no_numpy(self):
        from agent_eval.utils.embeddings import _NUMPY_AVAILABLE

        if not _NUMPY_AVAILABLE:
            pytest.skip("numpy not installed")

    def test_identical_vectors(self):
        self._skip_if_no_numpy()
        from agent_eval.utils.embeddings import cosine_similarity

        assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        self._skip_if_no_numpy()
        from agent_eval.utils.embeddings import cosine_similarity

        assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9

    def test_opposite_vectors(self):
        self._skip_if_no_numpy()
        from agent_eval.utils.embeddings import cosine_similarity

        assert abs(cosine_similarity([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-9

    def test_similar_vectors(self):
        self._skip_if_no_numpy()
        from agent_eval.utils.embeddings import cosine_similarity

        sim = cosine_similarity([1.0, 1.0], [1.0, 0.5])
        assert 0.9 < sim < 1.0

    def test_zero_vector_returns_zero(self):
        self._skip_if_no_numpy()
        from agent_eval.utils.embeddings import cosine_similarity

        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_high_dimensional(self):
        self._skip_if_no_numpy()
        from agent_eval.utils.embeddings import cosine_similarity

        a = [float(i) for i in range(100)]
        b = [float(i + 1) for i in range(100)]
        sim = cosine_similarity(a, b)
        assert 0.99 < sim <= 1.0


class TestMakeSimilarityFn:
    def test_basic_usage(self):
        from agent_eval.utils.embeddings import _NUMPY_AVAILABLE

        if not _NUMPY_AVAILABLE:
            pytest.skip("numpy not installed")

        from agent_eval.utils.embeddings import make_similarity_fn

        # Fake embedding function: just uses character counts
        def fake_embed(text):
            return [float(len(text)), float(text.count("a")), float(text.count("e"))]

        sim_fn = make_similarity_fn(fake_embed)
        result = sim_fn("hello", "hello")
        assert abs(result - 1.0) < 1e-9

    def test_different_texts(self):
        from agent_eval.utils.embeddings import _NUMPY_AVAILABLE

        if not _NUMPY_AVAILABLE:
            pytest.skip("numpy not installed")

        from agent_eval.utils.embeddings import make_similarity_fn

        def fake_embed(text):
            return [float(ord(c)) for c in text[:5].ljust(5)]

        sim_fn = make_similarity_fn(fake_embed)
        result = sim_fn("hello", "world")
        assert -1.0 <= result <= 1.0
