"""Embedding utilities — cosine similarity and vector helpers.

Ported from ``experiments/prompt_tunning/utils.py`` but made
framework-agnostic: only depends on numpy (no Azure/LangChain).

Users supply their own ``embed_fn`` for producing vectors.
"""

from __future__ import annotations

from typing import Callable

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


def _check_numpy() -> None:
    if not _NUMPY_AVAILABLE:
        raise ImportError("numpy is not installed. " "Install with: pip install numpy")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Accepts plain Python lists or numpy arrays.
    Returns a float in [-1, 1].
    """
    _check_numpy()
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    dot = float(np.dot(va, vb))
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if norm < 1e-12:
        return 0.0
    return dot / norm


def make_similarity_fn(
    embed_fn: Callable[[str], list[float]],
) -> Callable[[str, str], float]:
    """Create a SimilarityFn from an embedding function.

    Parameters
    ----------
    embed_fn
        A callable that maps text to a float vector.
        For example, ``openai.embeddings.create(...)`` or
        ``sentence_transformers.encode(...)``.

    Returns
    -------
    Callable[[str, str], float]
        A function that computes cosine similarity between two texts.
        Compatible with ``TuningLoop``'s ``similarity_fn`` parameter.

    Example
    -------
    ::

        from langchain_openai import AzureOpenAIEmbeddings
        embeddings = AzureOpenAIEmbeddings(...)

        sim_fn = make_similarity_fn(embeddings.embed_query)
        score = sim_fn("hello world", "hi there")
    """

    def similarity(text_a: str, text_b: str) -> float:
        vec_a = embed_fn(text_a)
        vec_b = embed_fn(text_b)
        return cosine_similarity(vec_a, vec_b)

    return similarity
