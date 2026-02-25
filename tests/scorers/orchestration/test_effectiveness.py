"""Tests for scorers.orchestration.effectiveness — OrchestrationScorer."""

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.protocols import Scorer
from agent_eval.core.score import Severity
from agent_eval.scorers.orchestration.effectiveness import OrchestrationScorer


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


def _msg(agent_name: str, content: str) -> Step:
    return Step(
        kind=StepKind.MESSAGE,
        agent_id=agent_name.lower(),
        agent_name=agent_name,
        content=content,
    )


def _tool(
    tool_name: str,
    succeeded: bool = True,
    agent_name: str = "",
    tool_args: dict | None = None,
) -> Step:
    return Step(
        kind=StepKind.TOOL_CALL,
        agent_id=agent_name.lower() if agent_name else "",
        agent_name=agent_name,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_succeeded=succeeded,
    )


class TestOrchestrationScorerBasics:
    def test_name(self):
        scorer = OrchestrationScorer()
        assert scorer.name == "orchestration"

    def test_protocol_compliance(self):
        scorer = OrchestrationScorer()
        assert isinstance(scorer, Scorer)

    def test_score_returns_five_dimensions(self):
        ep = _make_episode(steps=[_tool("test_tool")])
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        assert len(dims) == 5
        names = {d.name for d in dims}
        assert names == {
            "delegation_efficiency",
            "tool_strategy",
            "coordination_overhead",
            "recovery_effectiveness",
            "termination_quality",
        }

    def test_empty_episode(self):
        ep = _make_episode(steps=[])
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        assert all(d.value == 1.0 for d in dims)


class TestDelegationEfficiency:
    def test_all_agents_productive(self):
        ep = _make_episode(
            steps=[
                _msg("Agent1", "Analyzing the data thoroughly here"),
                _tool("get_data", agent_name="Agent2"),
            ]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        delegation = next(d for d in dims if d.name == "delegation_efficiency")
        assert delegation.value == 1.0

    def test_idle_agent(self):
        ep = _make_episode(
            steps=[
                _msg("Agent1", "Analyzing the data thoroughly here"),
                _msg("IdleAgent", "Ok"),  # Short message = not productive
                _tool("get_data", agent_name="Agent2"),
            ]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        delegation = next(d for d in dims if d.name == "delegation_efficiency")
        assert delegation.value < 1.0

    def test_idle_agent_issue(self):
        ep = _make_episode(
            steps=[
                _msg("Agent1", "Analyzing the data thoroughly here"),
                _msg("IdleAgent", "Ok"),
            ]
        )
        scorer = OrchestrationScorer()
        issues = scorer.detect_issues(ep)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any("idle agent" in i.description.lower() for i in warnings)


class TestToolStrategy:
    def test_diverse_tools_no_failures(self):
        ep = _make_episode(
            steps=[
                _tool("search_db"),
                _tool("call_api"),
                _tool("generate_chart"),
            ]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        strategy = next(d for d in dims if d.name == "tool_strategy")
        assert strategy.value == 1.0

    def test_retry_loop_penalty(self):
        ep = _make_episode(
            steps=[
                _tool("bad_tool", succeeded=False),
                _tool("bad_tool", succeeded=False),
                _tool("bad_tool", succeeded=False),
            ]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        strategy = next(d for d in dims if d.name == "tool_strategy")
        assert strategy.value < 1.0

    def test_retry_loop_issue(self):
        ep = _make_episode(
            steps=[
                _tool("bad_tool", succeeded=False),
                _tool("bad_tool", succeeded=False),
                _tool("bad_tool", succeeded=False),
            ]
        )
        scorer = OrchestrationScorer()
        issues = scorer.detect_issues(ep)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert any("tool retry loop" in i.description.lower() for i in errors)

    def test_redundant_calls_penalty(self):
        ep = _make_episode(
            steps=[
                _tool("search", tool_args={"q": "test"}),
                _tool("search", tool_args={"q": "test"}),
                _tool("search", tool_args={"q": "test"}),
                _tool("other_tool"),
            ]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        strategy = next(d for d in dims if d.name == "tool_strategy")
        assert strategy.value < 1.0

    def test_no_tool_calls(self):
        ep = _make_episode(steps=[_msg("Agent1", "Just talking")])
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        strategy = next(d for d in dims if d.name == "tool_strategy")
        assert strategy.value == 1.0


class TestCoordinationOverhead:
    def test_all_productive(self):
        ep = _make_episode(
            steps=[
                _tool("search"),
                _tool("analyze"),
                Step(
                    kind=StepKind.FACT_CHECK,
                    agent_id="fc",
                    agent_name="FactChecker",
                    metadata={"verdict": "PASS"},
                ),
            ]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        overhead = next(d for d in dims if d.name == "coordination_overhead")
        assert overhead.value == 1.0

    def test_high_overhead(self):
        steps = [_msg("Agent1", "Chatting") for _ in range(8)]
        steps.append(_tool("one_useful_tool"))
        ep = _make_episode(steps=steps)
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        overhead = next(d for d in dims if d.name == "coordination_overhead")
        assert overhead.value < 0.3

    def test_high_overhead_issue(self):
        steps = [_msg("Agent1", "Chatting") for _ in range(8)]
        steps.append(_tool("one_tool"))
        ep = _make_episode(steps=steps)
        scorer = OrchestrationScorer()
        issues = scorer.detect_issues(ep)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any("coordination overhead" in i.description.lower() for i in warnings)


class TestRecoveryEffectiveness:
    def test_no_failures(self):
        ep = _make_episode(
            steps=[_tool("t1"), _tool("t2")]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        recovery = next(d for d in dims if d.name == "recovery_effectiveness")
        assert recovery.value == 1.0

    def test_failure_with_recovery(self):
        ep = _make_episode(
            steps=[
                _tool("t1", succeeded=False),
                _tool("t2", succeeded=True),
            ]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        recovery = next(d for d in dims if d.name == "recovery_effectiveness")
        assert recovery.value == 1.0

    def test_failure_without_recovery(self):
        ep = _make_episode(
            steps=[
                _tool("t1", succeeded=True),
                _tool("t2", succeeded=False),
            ]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        recovery = next(d for d in dims if d.name == "recovery_effectiveness")
        assert recovery.value == 0.0

    def test_unrecovered_failure_issue(self):
        ep = _make_episode(
            steps=[
                _tool("t1", succeeded=True),
                _tool("t2", succeeded=False),
            ]
        )
        scorer = OrchestrationScorer()
        issues = scorer.detect_issues(ep)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any("unrecovered" in i.description.lower() for i in warnings)

    def test_partial_recovery(self):
        ep = _make_episode(
            steps=[
                _tool("t1", succeeded=False),
                _tool("t2", succeeded=True),
                _tool("t3", succeeded=False),
                # no recovery after t3
            ]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        recovery = next(d for d in dims if d.name == "recovery_effectiveness")
        assert recovery.value == 0.5


class TestTerminationQuality:
    def test_clean_termination(self):
        ep = _make_episode(
            steps=[
                _tool("search"),
                _tool("analyze"),
            ]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        termination = next(d for d in dims if d.name == "termination_quality")
        assert termination.value == 1.0

    def test_wasted_steps_after_answer(self):
        ep = _make_episode(
            steps=[
                _tool("search"),
                _msg("Agent1", "Here is the final answer and conclusion"),
                _msg("Agent1", "Just chatting now"),
                _msg("Agent1", "More unnecessary talk"),
            ]
        )
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        termination = next(d for d in dims if d.name == "termination_quality")
        assert termination.value < 1.0

    def test_no_steps(self):
        ep = _make_episode(steps=[])
        scorer = OrchestrationScorer()
        dims = scorer.score(ep)
        termination = next(d for d in dims if d.name == "termination_quality")
        assert termination.value == 1.0
