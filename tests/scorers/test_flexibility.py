"""Tests for scorer flexibility features: configurable thresholds, framework
agent filtering, and substantive message handling."""

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Issue, Severity
from agent_eval.scorers.orchestration.effectiveness import OrchestrationScorer
from agent_eval.scorers.rules.deduction import RuleBasedScorer
from agent_eval.scorers.rules.issue_detector import IssueDetectorScorer


def _make_episode(**kwargs) -> Episode:
    defaults = {
        "episode_id": "test",
        "steps": [],
        "source_framework": "test",
        "final_answer": "A detailed answer with enough content." * 5,
        "metadata": {},
    }
    defaults.update(kwargs)
    return Episode(**defaults)


def _msg(agent_name: str, content: str, framework: bool = False) -> Step:
    meta = {}
    if framework:
        meta["framework_agent"] = True
    return Step(
        kind=StepKind.MESSAGE,
        agent_id=agent_name.lower(),
        agent_name=agent_name,
        content=content,
        metadata=meta,
    )


def _tool(tool_name: str, agent_name: str = "", args: dict | None = None,
          succeeded: bool = True) -> Step:
    return Step(
        kind=StepKind.TOOL_CALL,
        agent_id=agent_name.lower() if agent_name else "",
        agent_name=agent_name,
        tool_name=tool_name,
        tool_args=args,
        tool_succeeded=succeeded,
    )


# ------------------------------------------------------------------
# IssueDetectorScorer flexibility
# ------------------------------------------------------------------


class TestStallDetectionWithArgs:
    def test_same_tool_different_args_no_stall(self):
        """Calling the same tool with different arguments should NOT be a stall."""
        ep = _make_episode(
            steps=[
                _tool("search", args={"q": "revenue 2024"}),
                _tool("search", args={"q": "revenue 2025"}),
                _tool("search", args={"q": "profit 2024"}),
                _tool("search", args={"q": "profit 2025"}),
                _tool("search", args={"q": "costs 2024"}),
            ]
        )
        scorer = IssueDetectorScorer()
        issues = scorer.detect_issues(ep)
        stall_issues = [
            i for i in issues if "stall" in i.description.lower()
        ]
        assert len(stall_issues) == 0

    def test_same_tool_same_args_is_stall(self):
        """Calling the same tool with identical arguments IS a stall."""
        ep = _make_episode(
            steps=[
                _tool("search", args={"q": "same"}),
                _tool("search", args={"q": "same"}),
                _tool("search", args={"q": "same"}),
                _tool("search", args={"q": "same"}),
                _tool("search", args={"q": "same"}),
            ]
        )
        scorer = IssueDetectorScorer()
        issues = scorer.detect_issues(ep)
        stall_issues = [
            i for i in issues if "stall" in i.description.lower()
        ]
        assert len(stall_issues) > 0

    def test_configurable_stall_threshold(self):
        """Higher stall threshold means fewer stall detections."""
        ep = _make_episode(
            steps=[
                _tool("search", args={"q": "same"}),
                _tool("search", args={"q": "same"}),
                _tool("search", args={"q": "same"}),
                _tool("search", args={"q": "same"}),
                _tool("search", args={"q": "same"}),
            ]
        )
        # Default threshold (3) should detect stalls
        strict = IssueDetectorScorer(stall_threshold=3)
        assert any("stall" in i.description.lower() for i in strict.detect_issues(ep))

        # High threshold should not
        lenient = IssueDetectorScorer(stall_threshold=10)
        assert not any("stall" in i.description.lower() for i in lenient.detect_issues(ep))


class TestFrameworkAgentExclusion:
    def test_framework_agents_excluded_from_imbalance(self):
        """Framework agents should not trigger turn imbalance warnings."""
        ep = _make_episode(
            steps=[
                _msg("SelectorGroupChatManager", "", framework=True),
                _msg("SelectorGroupChatManager", "", framework=True),
                _msg("SelectorGroupChatManager", "", framework=True),
                _msg("SelectorGroupChatManager", "", framework=True),
                _msg("SelectorGroupChatManager", "", framework=True),
                _msg("SelectorGroupChatManager", "", framework=True),
                _msg("Worker1", "Analyzing data in detail"),
                _msg("Worker2", "Processing results thoroughly"),
                _tool("search"),
            ]
        )
        scorer = IssueDetectorScorer()
        issues = scorer.detect_issues(ep)
        imbalance = [i for i in issues if "imbalance" in i.description.lower()]
        assert len(imbalance) == 0

    def test_framework_agents_excluded_from_delegation_count(self):
        """Only non-framework agents should count for delegation check."""
        ep = _make_episode(
            steps=[
                _msg("SelectorGroupChatManager", "", framework=True),
                _msg("MagenticOneOrchestrator", "routing", framework=True),
                _msg("Worker1", "Doing the real work here"),
                _tool("search"),
            ]
        )
        scorer = IssueDetectorScorer()
        issues = scorer.detect_issues(ep)
        delegation = [i for i in issues if "delegation" in i.description.lower()]
        # Only 1 worker agent → should flag insufficient delegation
        assert len(delegation) > 0


# ------------------------------------------------------------------
# OrchestrationScorer flexibility
# ------------------------------------------------------------------


class TestOrchestrationFrameworkFiltering:
    def test_framework_agents_not_idle(self):
        """Framework agents should not appear in idle agent list."""
        ep = _make_episode(
            steps=[
                _msg("SelectorGroupChatManager", "", framework=True),
                _msg("Worker1", "Analyzing data in detail"),
                _tool("search", agent_name="Worker1"),
            ]
        )
        scorer = OrchestrationScorer()
        issues = scorer.detect_issues(ep)
        idle = [i for i in issues if "idle" in i.description.lower()]
        idle_names = [i.description for i in idle]
        assert not any("SelectorGroupChatManager" in d for d in idle_names)

    def test_framework_agents_not_in_delegation(self):
        """Framework agents should not lower delegation efficiency."""
        ep = _make_episode(
            steps=[
                _msg("SelectorGroupChatManager", "", framework=True),
                _msg("Orchestrator", "", framework=True),
                _msg("Worker1", "Analyzing data in detail"),
                _tool("search", agent_name="Worker1"),
                _msg("Worker2", "Processing the results here"),
                _tool("analyze", agent_name="Worker2"),
            ]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        delegation = next(d for d in dims if d.name == "delegation_efficiency")
        assert delegation.value == 1.0  # Both workers are productive


class TestSubstantiveMessagesProductive:
    def test_long_messages_count_as_productive(self):
        """Substantive messages should improve coordination overhead score."""
        steps = [
            _msg("Agent1", "Analyzing the financial data thoroughly"),
            _msg("Agent2", "Based on that analysis, here are my findings on revenue"),
            _tool("search"),
        ]
        ep = _make_episode(steps=steps)
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        overhead = next(d for d in dims if d.name == "coordination_overhead")
        # All 3 steps are productive (2 long messages + 1 tool)
        assert overhead.value == 1.0

    def test_configurable_message_threshold(self):
        """Custom min_productive_message_len changes what counts."""
        steps = [
            _msg("Agent1", "Short msg"),  # 9 chars
            _tool("search"),
        ]
        ep = _make_episode(steps=steps)

        # Default: 20 chars → message is not productive → 1/2
        default_scorer = OrchestrationScorer()
        dims = default_scorer.score(ep)
        overhead = next(d for d in dims if d.name == "coordination_overhead")
        assert overhead.value == 0.5

        # Low threshold: 5 chars → message IS productive → 2/2
        lenient_scorer = OrchestrationScorer(min_productive_message_len=5)
        dims = lenient_scorer.score(ep)
        overhead = next(d for d in dims if d.name == "coordination_overhead")
        assert overhead.value == 1.0


# ------------------------------------------------------------------
# RuleBasedScorer flexibility
# ------------------------------------------------------------------


class TestConfigurableDeductions:
    def test_custom_severity_weights(self):
        ep = _make_episode()
        issues = [Issue(Severity.CRITICAL, "Test", "Critical")]
        # Default: -25
        default_scorer = RuleBasedScorer()
        dims = default_scorer.score_with_issues(ep, issues)
        assert dims[0].value == 75.0

        # Custom: -10
        light_scorer = RuleBasedScorer(critical_weight=10)
        dims = light_scorer.score_with_issues(ep, issues)
        assert dims[0].value == 90.0

    def test_custom_grade_thresholds(self):
        scorer = RuleBasedScorer(
            grade_thresholds={95: "S", 80: "A", 60: "B", 40: "C"}
        )
        assert scorer.get_grade(96) == "S"
        assert scorer.get_grade(85) == "A"
        assert scorer.get_grade(65) == "B"
        assert scorer.get_grade(45) == "C"
        assert scorer.get_grade(30) == "F"

    def test_custom_bonus_thresholds(self):
        ep = _make_episode(
            final_answer="A" * 200,  # Below default 500, above custom 100
            steps=[
                _tool("t1", succeeded=True),
                _tool("t2", succeeded=True),
            ],
        )
        # Add issues to bring base below 100 so bonus difference is visible
        issues = [Issue(Severity.WARNING, "Test", "W1")]

        # Default: 500 char threshold → no answer bonus → 95
        default_scorer = RuleBasedScorer()
        dims = default_scorer.score_with_issues(ep, issues)
        default_score = dims[0].value

        # Custom: 100 char threshold → gets answer bonus → 100
        custom_scorer = RuleBasedScorer(answer_length_threshold=100)
        dims = custom_scorer.score_with_issues(ep, issues)
        custom_score = dims[0].value

        assert custom_score > default_score
