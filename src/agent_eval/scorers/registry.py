"""Scorer plugin discovery and registry.

Provides a central registry for Scorer implementations so they can
be looked up by name (e.g. from CLI arguments or config files).

Usage::

    from agent_eval.scorers.registry import ScorerRegistry

    registry = ScorerRegistry()
    registry.register("numeric", NumericConsistencyScorer)
    registry.register("issues", IssueDetectorScorer)

    scorer = registry.get("numeric", tolerance=0.1)

Built-in scorers are auto-registered on first access.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ScorerRegistry:
    """Registry mapping scorer names to factory callables.

    Each entry is a callable that accepts keyword arguments and returns
    a Scorer-protocol-compatible instance.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., Any]] = {}
        self._auto_registered = False

    def register(
        self, name: str, factory: Callable[..., Any], overwrite: bool = False
    ) -> None:
        """Register a scorer factory under the given name.

        Parameters
        ----------
        name
            Lookup key (e.g. ``"numeric"``, ``"issue_detector"``).
        factory
            A callable (typically a class) that returns a Scorer instance.
        overwrite
            If False (default), raises ValueError on duplicate names.
        """
        if name in self._registry and not overwrite:
            raise ValueError(
                f"Scorer '{name}' is already registered. "
                "Use overwrite=True to replace it."
            )
        self._registry[name] = factory

    def get(self, name: str, **kwargs: Any) -> Any:
        """Look up a scorer by name and instantiate it.

        Parameters
        ----------
        name
            Registered scorer name.
        **kwargs
            Passed to the scorer factory/constructor.

        Returns
        -------
        Scorer instance.

        Raises
        ------
        KeyError
            If the name is not registered.
        """
        self._ensure_builtins()
        if name not in self._registry:
            raise KeyError(
                f"Scorer '{name}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name](**kwargs)

    def list_scorers(self) -> list[str]:
        """Return all registered scorer names."""
        self._ensure_builtins()
        return sorted(self._registry.keys())

    def _ensure_builtins(self) -> None:
        """Lazily register built-in scorers on first access."""
        if self._auto_registered:
            return
        self._auto_registered = True
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register the built-in scorer implementations."""
        # Core scorers (always available)
        from agent_eval.scorers.numeric.consistency import NumericConsistencyScorer
        from agent_eval.scorers.rules.deduction import RuleBasedScorer
        from agent_eval.scorers.rules.issue_detector import IssueDetectorScorer

        self._registry.setdefault("numeric", NumericConsistencyScorer)
        self._registry.setdefault("issue_detector", IssueDetectorScorer)
        self._registry.setdefault("rule_based", RuleBasedScorer)

        # LLM Judge (requires an llm_fn argument)
        from agent_eval.scorers.llm_judge.judge import LLMJudgeScorer

        self._registry.setdefault("llm_judge", LLMJudgeScorer)

        # Optional: DeepEval (only if installed)
        try:
            from agent_eval.scorers.deepeval.wrapper import DeepEvalScorer

            self._registry.setdefault("deepeval", DeepEvalScorer)
        except ImportError:
            pass

        # Optional: Ragas (only if installed)
        try:
            from agent_eval.scorers.ragas.wrapper import RagasScorer

            self._registry.setdefault("ragas", RagasScorer)
        except ImportError:
            pass


# Module-level singleton for convenience
default_registry = ScorerRegistry()
