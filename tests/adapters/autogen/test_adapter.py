"""Tests for adapters.autogen.adapter — AutoGenAdapter."""

import tempfile
from pathlib import Path

import pytest

from agent_eval.adapters.autogen.adapter import AutoGenAdapter
from agent_eval.core.exceptions import AdapterError
from agent_eval.core.models import StepKind


AGENT_NAMES = [
    "SalesNegotiator",
    "FinanceExpert",
    "CustomerResearcher",
    "DataVisualiser",
]


class TestAutoGenAdapterInit:
    def test_framework_name(self):
        adapter = AutoGenAdapter()
        assert adapter.framework_name == "autogen"

    def test_custom_agent_names(self):
        adapter = AutoGenAdapter(agent_names=["Agent1", "Agent2"])
        assert adapter.agent_names == ["Agent1", "Agent2"]


class TestAutoGenAdapterLoadEpisode:
    def test_load_from_fixture(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        assert episode.source_framework == "autogen"
        assert episode.episode_id  # should be a UUID
        assert len(episode.steps) > 0

    def test_load_from_directory(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        # Pass the directory containing event.txt
        episode = adapter.load_episode(str(sample_log_path.parent))
        assert len(episode.steps) > 0

    def test_step_kinds_present(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        kinds = {s.kind for s in episode.steps}
        assert StepKind.MESSAGE in kinds
        assert StepKind.TOOL_CALL in kinds
        assert StepKind.LLM_CALL in kinds
        assert StepKind.FACT_CHECK in kinds

    def test_tool_calls_detected(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        tool_steps = episode.steps_by_kind(StepKind.TOOL_CALL)
        assert len(tool_steps) == 3
        tool_names = {s.tool_name for s in tool_steps}
        assert "get_product_finances" in tool_names
        assert "ask_web" in tool_names
        assert "plot_bar_chart" in tool_names

    def test_tool_success_detection(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        tool_steps = episode.steps_by_kind(StepKind.TOOL_CALL)
        assert all(s.tool_succeeded is True for s in tool_steps)

    def test_agent_names_resolved(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        msg_steps = episode.steps_by_kind(StepKind.MESSAGE)
        agent_names = {s.agent_name for s in msg_steps}
        assert "FinanceExpert" in agent_names
        assert "CustomerResearcher" in agent_names
        assert "DataVisualiser" in agent_names

    def test_timestamps_extracted(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        assert episode.started_at is not None
        assert episode.ended_at is not None
        assert episode.duration_seconds > 0

    def test_final_answer_extracted(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        assert episode.final_answer is not None
        assert "283M" in episode.final_answer

    def test_fact_check_steps(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episode = adapter.load_episode(str(sample_log_path))

        fc_steps = episode.steps_by_kind(StepKind.FACT_CHECK)
        assert len(fc_steps) == 1
        assert fc_steps[0].metadata["verdict"] == "PASS"

    def test_missing_path_raises(self):
        adapter = AutoGenAdapter()
        with pytest.raises(AdapterError, match="Could not find event.txt"):
            adapter.load_episode("/nonexistent/path")


class TestAutoGenAdapterLoadEpisodes:
    def test_load_from_fixture_dir(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episodes = adapter.load_episodes(str(sample_log_path.parent))
        assert len(episodes) == 1

    def test_load_single_file_fallback(self, sample_log_path):
        adapter = AutoGenAdapter(agent_names=AGENT_NAMES)
        episodes = adapter.load_episodes(str(sample_log_path))
        assert len(episodes) == 1

    def test_load_from_temp_dir_with_multiple_logs(self):
        adapter = AutoGenAdapter()

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                log_dir = Path(tmpdir) / f"log-{i}"
                log_dir.mkdir()
                (log_dir / "event.txt").write_text(
                    f'{{"type": "Message", "sender": "Agent{i}", "content": "msg"}}'
                )

            episodes = adapter.load_episodes(tmpdir)
            assert len(episodes) == 3
