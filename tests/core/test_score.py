"""Tests for core.score — ScoreVector, ScoreDimension, Issue, Severity."""

from agent_eval.core.score import Issue, ScoreDimension, ScoreVector, Severity


class TestSeverity:
    def test_enum_values(self):
        assert Severity.CRITICAL.value == "CRITICAL"
        assert Severity.ERROR.value == "ERROR"
        assert Severity.WARNING.value == "WARNING"
        assert Severity.INFO.value == "INFO"


class TestIssue:
    def test_minimal_issue(self):
        issue = Issue(
            severity=Severity.WARNING,
            category="Test",
            description="test issue",
        )
        assert issue.line_number is None
        assert issue.context is None

    def test_full_issue(self):
        issue = Issue(
            severity=Severity.CRITICAL,
            category="Error Detection",
            description="Exception found",
            line_number=42,
            context="ValueError: invalid",
        )
        assert issue.line_number == 42
        assert issue.context == "ValueError: invalid"


class TestScoreDimension:
    def test_normalized_value(self):
        dim = ScoreDimension(name="test", value=75.0, max_value=100.0)
        assert dim.normalized == 0.75

    def test_normalized_capped_at_one(self):
        dim = ScoreDimension(name="test", value=110.0, max_value=100.0)
        assert dim.normalized == 1.0

    def test_normalized_zero_max(self):
        dim = ScoreDimension(name="test", value=5.0, max_value=0.0)
        assert dim.normalized == 0.0

    def test_default_max_value(self):
        dim = ScoreDimension(name="test", value=0.8)
        assert dim.max_value == 1.0
        assert dim.normalized == 0.8


class TestScoreVector:
    def test_overall_with_dimensions(self):
        sv = ScoreVector(
            episode_id="test",
            dimensions=[
                ScoreDimension(name="a", value=0.8),
                ScoreDimension(name="b", value=0.6),
            ],
        )
        assert sv.overall == pytest.approx(0.7)

    def test_overall_empty(self):
        sv = ScoreVector(episode_id="test")
        assert sv.overall == 0.0

    def test_dimension_by_name(self):
        sv = ScoreVector(
            episode_id="test",
            dimensions=[
                ScoreDimension(name="accuracy", value=0.9),
                ScoreDimension(name="speed", value=0.5),
            ],
        )
        assert sv.dimension_by_name("accuracy").value == 0.9
        assert sv.dimension_by_name("missing") is None

    def test_to_dict(self):
        sv = ScoreVector(
            episode_id="ep-1",
            dimensions=[ScoreDimension(name="test", value=0.8, source="scorer1")],
            issues=[
                Issue(
                    severity=Severity.WARNING,
                    category="Test",
                    description="test warning",
                )
            ],
        )
        d = sv.to_dict()
        assert d["episode_id"] == "ep-1"
        assert len(d["dimensions"]) == 1
        assert d["dimensions"][0]["name"] == "test"
        assert len(d["issues"]) == 1
        assert d["issues"][0]["severity"] == "WARNING"


import pytest
