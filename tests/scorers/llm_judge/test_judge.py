"""Tests for scorers.llm_judge — LLMJudgeScorer."""

import json

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Severity
from agent_eval.scorers.llm_judge.judge import LLMJudgeScorer
from agent_eval.scorers.llm_judge.prompts import (
    DEFAULT_DIMENSIONS,
    build_transcript,
    format_multi_prompt,
    format_single_prompt,
)


# --- Mock LLM functions ---


def _mock_llm_batch(system_prompt: str, user_prompt: str) -> str:
    """Return a valid multi-dimension JSON response."""
    evaluations = [
        {"dimension": name, "score": 0.8, "justification": "Good performance."}
        for name in DEFAULT_DIMENSIONS
    ]
    return json.dumps({"evaluations": evaluations})


def _mock_llm_single(system_prompt: str, user_prompt: str) -> str:
    """Return a valid single-dimension JSON response."""
    return json.dumps(
        {
            "dimension": "relevance",
            "score": 0.75,
            "justification": "Mostly relevant.",
        }
    )


def _mock_llm_malformed(system_prompt: str, user_prompt: str) -> str:
    """Return garbage text."""
    return "I can't evaluate this, sorry!"


def _mock_llm_with_markdown(system_prompt: str, user_prompt: str) -> str:
    """Return JSON wrapped in markdown code fence."""
    evaluations = [
        {"dimension": name, "score": 0.9, "justification": "Excellent."}
        for name in DEFAULT_DIMENSIONS
    ]
    return f"```json\n{json.dumps({'evaluations': evaluations})}\n```"


def _make_episode() -> Episode:
    return Episode(
        episode_id="test-judge",
        steps=[
            Step(
                kind=StepKind.MESSAGE,
                agent_id="a1",
                agent_name="Agent1",
                content="Analyzing financial data",
            ),
            Step(
                kind=StepKind.TOOL_CALL,
                agent_id="",
                agent_name="",
                tool_name="get_data",
                tool_result={"revenue": 100000},
                tool_succeeded=True,
            ),
            Step(
                kind=StepKind.FACT_CHECK,
                agent_id="a1",
                agent_name="Agent1",
                metadata={"verdict": "PASS"},
            ),
        ],
        source_framework="test",
        final_answer="The revenue is 100,000 based on tool data.",
    )


class TestLLMJudgeScorerBatch:
    def test_name(self):
        scorer = LLMJudgeScorer(llm_fn=_mock_llm_batch)
        assert scorer.name == "llm_judge"

    def test_score_returns_all_dimensions(self):
        scorer = LLMJudgeScorer(llm_fn=_mock_llm_batch)
        ep = _make_episode()
        dims = scorer.score(ep)
        assert len(dims) == len(DEFAULT_DIMENSIONS)
        for dim in dims:
            assert dim.value == 0.8
            assert dim.source == "llm_judge"

    def test_detect_issues_low_scores(self):
        def low_scorer(sys_p, usr_p):
            evals = [
                {"dimension": name, "score": 0.2, "justification": "Bad."}
                for name in DEFAULT_DIMENSIONS
            ]
            return json.dumps({"evaluations": evals})

        scorer = LLMJudgeScorer(llm_fn=low_scorer, low_score_threshold=0.4)
        ep = _make_episode()
        issues = scorer.detect_issues(ep)
        assert len(issues) == len(DEFAULT_DIMENSIONS)
        assert all(i.severity == Severity.WARNING for i in issues)

    def test_detect_issues_no_issues_on_high_scores(self):
        scorer = LLMJudgeScorer(llm_fn=_mock_llm_batch, low_score_threshold=0.4)
        ep = _make_episode()
        issues = scorer.detect_issues(ep)
        assert len(issues) == 0

    def test_malformed_response_fallback(self):
        scorer = LLMJudgeScorer(llm_fn=_mock_llm_malformed)
        ep = _make_episode()
        dims = scorer.score(ep)
        # Should return zero-scored fallback dimensions
        assert len(dims) == len(DEFAULT_DIMENSIONS)
        assert all(d.value == 0.0 for d in dims)

    def test_markdown_json_extraction(self):
        scorer = LLMJudgeScorer(llm_fn=_mock_llm_with_markdown)
        ep = _make_episode()
        dims = scorer.score(ep)
        assert len(dims) == len(DEFAULT_DIMENSIONS)
        assert all(d.value == 0.9 for d in dims)


class TestLLMJudgeScorerIndividual:
    def test_individual_mode(self):
        scorer = LLMJudgeScorer(
            llm_fn=_mock_llm_single,
            batch_dimensions=False,
            dimensions={"relevance": "Is it relevant?"},
        )
        ep = _make_episode()
        dims = scorer.score(ep)
        assert len(dims) == 1
        assert dims[0].name == "relevance"
        assert dims[0].value == 0.75

    def test_individual_mode_failure_graceful(self):
        scorer = LLMJudgeScorer(
            llm_fn=_mock_llm_malformed,
            batch_dimensions=False,
            dimensions={"relevance": "Is it relevant?"},
        )
        ep = _make_episode()
        dims = scorer.score(ep)
        assert len(dims) == 1
        assert dims[0].value == 0.0


class TestPromptHelpers:
    def test_build_transcript_truncation(self):
        lines = ["Line " * 100] * 100
        result = build_transcript(lines, max_chars=200)
        assert len(result) <= 220  # 200 + "[truncated]"
        assert "truncated" in result

    def test_format_single_prompt(self):
        prompt = format_single_prompt(
            "relevance", "Is it relevant?", "Agent: hello", "The answer"
        )
        assert "relevance" in prompt
        assert "Agent: hello" in prompt
        assert "The answer" in prompt

    def test_format_multi_prompt(self):
        dims = {"relevance": "Is it relevant?", "coherence": "Is it coherent?"}
        prompt = format_multi_prompt(dims, "Agent: hello", "The answer")
        assert "relevance" in prompt
        assert "coherence" in prompt
        assert "Agent: hello" in prompt

    def test_custom_dimensions(self):
        scorer = LLMJudgeScorer(
            llm_fn=_mock_llm_batch,
            dimensions={"custom_dim": "Custom description"},
        )
        assert "custom_dim" in scorer.dimensions
