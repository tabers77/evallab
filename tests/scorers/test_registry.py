"""Tests for scorers.registry — ScorerRegistry."""

import pytest

from agent_eval.scorers.registry import ScorerRegistry, default_registry


class TestScorerRegistry:
    def test_register_and_get(self):
        reg = ScorerRegistry()

        class FakeScorer:
            def __init__(self, x=1):
                self.x = x

        reg.register("fake", FakeScorer)
        scorer = reg.get("fake", x=42)
        assert scorer.x == 42

    def test_duplicate_register_raises(self):
        reg = ScorerRegistry()
        reg.register("a", lambda: None)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("a", lambda: None)

    def test_overwrite_allowed(self):
        reg = ScorerRegistry()
        reg.register("a", lambda: "first")
        reg.register("a", lambda: "second", overwrite=True)
        assert reg.get("a") == "second"

    def test_get_unknown_raises(self):
        reg = ScorerRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_list_scorers_includes_builtins(self):
        reg = ScorerRegistry()
        names = reg.list_scorers()
        assert "numeric" in names
        assert "issue_detector" in names
        assert "rule_based" in names
        assert "llm_judge" in names

    def test_builtin_numeric_instantiation(self):
        reg = ScorerRegistry()
        scorer = reg.get("numeric", tolerance=0.1)
        assert scorer.tolerance == 0.1

    def test_builtin_issue_detector_instantiation(self):
        reg = ScorerRegistry()
        scorer = reg.get("issue_detector", max_llm_calls=50)
        assert scorer.max_llm_calls == 50


class TestDefaultRegistry:
    def test_default_registry_works(self):
        names = default_registry.list_scorers()
        assert len(names) >= 3
