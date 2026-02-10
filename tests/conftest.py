"""Shared fixtures and Episode factories for agent_eval tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_eval.core.models import Episode, Step, StepKind

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_LOG = FIXTURES_DIR / "sample_log" / "event.txt"


@pytest.fixture
def sample_log_path() -> Path:
    return SAMPLE_LOG


@pytest.fixture
def sample_log_content() -> str:
    return SAMPLE_LOG.read_text(encoding="utf-8")


@pytest.fixture
def minimal_episode() -> Episode:
    """An episode with one message, one tool call, and a final answer."""
    return Episode(
        episode_id="test-001",
        steps=[
            Step(
                kind=StepKind.MESSAGE,
                agent_id="Agent1_abc",
                agent_name="Agent1",
                content="Analyzing data",
                timestamp=datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
            ),
            Step(
                kind=StepKind.TOOL_CALL,
                agent_id="",
                agent_name="",
                tool_name="get_data",
                tool_args={"customer": "TestCorp"},
                tool_result={"revenue": 100000.0},
                tool_succeeded=True,
                timestamp=datetime(2026, 1, 15, 14, 1, 0, tzinfo=timezone.utc),
            ),
        ],
        source_framework="autogen",
        final_answer="The revenue is 100,000.",
        started_at=datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 15, 14, 2, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def multi_agent_episode() -> Episode:
    """An episode with multiple agents and tool calls."""
    return Episode(
        episode_id="test-002",
        steps=[
            Step(
                kind=StepKind.MESSAGE,
                agent_id="SalesNegotiator_abc",
                agent_name="SalesNegotiator",
                content="Planning approach",
            ),
            Step(
                kind=StepKind.MESSAGE,
                agent_id="FinanceExpert_def",
                agent_name="FinanceExpert",
                content="Analyzing finances",
            ),
            Step(
                kind=StepKind.TOOL_CALL,
                agent_id="",
                agent_name="",
                tool_name="get_product_finances",
                tool_result={"REVENUE": 283399382.94},
                tool_succeeded=True,
            ),
            Step(
                kind=StepKind.MESSAGE,
                agent_id="CustomerResearcher_ghi",
                agent_name="CustomerResearcher",
                content="Researching customer",
            ),
            Step(
                kind=StepKind.TOOL_CALL,
                agent_id="",
                agent_name="",
                tool_name="ask_web",
                tool_result="Market data found",
                tool_succeeded=True,
            ),
            Step(
                kind=StepKind.TOOL_CALL,
                agent_id="",
                agent_name="",
                tool_name="plot_bar_chart",
                tool_result={"chart_url": "https://example.com/chart"},
                tool_succeeded=True,
            ),
            Step(
                kind=StepKind.LLM_CALL,
                agent_id="",
                agent_name="",
                model="gpt-4o",
            ),
            Step(
                kind=StepKind.LLM_CALL,
                agent_id="",
                agent_name="",
                model="gpt-4o",
            ),
            Step(
                kind=StepKind.FACT_CHECK,
                agent_id="FinanceExpert",
                agent_name="FinanceExpert",
                content="Revenue verified",
                metadata={"verdict": "PASS", "reasoning": ["OK"]},
            ),
        ],
        source_framework="autogen",
        final_answer=(
            "<ANSWER>: Based on analysis, TestCorp revenue is 283M EUR "
            "with strong growth trajectory and positive market indicators. "
            "This comprehensive analysis covers financial data, customer "
            "research, and visualization of trends over multiple periods. "
            "We recommend a negotiation strategy that leverages our strong "
            "position in the Nordic market expansion. " * 3
        ),
        started_at=datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 15, 14, 3, 0, tzinfo=timezone.utc),
    )
