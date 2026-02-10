"""Tests for adapters.langgraph.checkpoint — checkpoint reader."""

import json
from pathlib import Path

import pytest

from agent_eval.adapters.langgraph.checkpoint import (
    checkpoint_to_episode,
    checkpoints_to_episodes,
    latest_checkpoint_to_episode,
    load_checkpoints,
)
from agent_eval.core.exceptions import AdapterError
from agent_eval.core.models import StepKind


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
SAMPLE_TRACE = FIXTURES_DIR / "sample_langgraph_trace.json"


class TestLoadCheckpoints:
    def test_load_from_fixture(self):
        checkpoints = load_checkpoints(str(SAMPLE_TRACE))
        assert len(checkpoints) == 3

    def test_missing_file_raises(self):
        with pytest.raises(AdapterError, match="not found"):
            load_checkpoints("/nonexistent/trace.json")

    def test_no_checkpoints_key_raises(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text('{"events": []}')
        with pytest.raises(AdapterError, match="No 'checkpoints'"):
            load_checkpoints(str(f))


class TestCheckpointToEpisode:
    def test_first_checkpoint(self):
        checkpoints = load_checkpoints(str(SAMPLE_TRACE))
        ep = checkpoint_to_episode(checkpoints[0])
        assert ep.source_framework == "langgraph"
        assert ep.episode_id == "cp-001"
        # First checkpoint: only user message
        messages = ep.steps_by_kind(StepKind.MESSAGE)
        assert len(messages) >= 1
        assert messages[0].agent_name == "user"

    def test_middle_checkpoint(self):
        checkpoints = load_checkpoints(str(SAMPLE_TRACE))
        ep = checkpoint_to_episode(checkpoints[1])
        # Middle checkpoint: user + assistant tool_call + tool result
        tool_calls = ep.steps_by_kind(StepKind.TOOL_CALL)
        tool_results = ep.steps_by_kind(StepKind.TOOL_RESULT)
        assert len(tool_calls) >= 1
        assert len(tool_results) >= 1

    def test_final_checkpoint(self):
        checkpoints = load_checkpoints(str(SAMPLE_TRACE))
        ep = checkpoint_to_episode(checkpoints[2])
        assert ep.final_answer is not None
        assert "283M" in ep.final_answer or "283" in ep.final_answer

    def test_task_extracted(self):
        checkpoints = load_checkpoints(str(SAMPLE_TRACE))
        ep = checkpoint_to_episode(checkpoints[0])
        assert ep.task_description is not None
        assert "revenue" in ep.task_description.lower()

    def test_custom_episode_id(self):
        checkpoints = load_checkpoints(str(SAMPLE_TRACE))
        ep = checkpoint_to_episode(checkpoints[0], episode_id="custom-id")
        assert ep.episode_id == "custom-id"


class TestCheckpointsToEpisodes:
    def test_converts_all(self):
        checkpoints = load_checkpoints(str(SAMPLE_TRACE))
        episodes = checkpoints_to_episodes(checkpoints)
        assert len(episodes) == 3
        # Each episode should have progressively more steps
        step_counts = [len(ep.steps) for ep in episodes]
        assert step_counts[0] <= step_counts[1] <= step_counts[2]


class TestLatestCheckpointToEpisode:
    def test_returns_last(self):
        checkpoints = load_checkpoints(str(SAMPLE_TRACE))
        ep = latest_checkpoint_to_episode(checkpoints)
        assert ep.episode_id == "cp-003"
        assert ep.final_answer is not None

    def test_empty_raises(self):
        with pytest.raises(AdapterError, match="Empty checkpoint"):
            latest_checkpoint_to_episode([])
