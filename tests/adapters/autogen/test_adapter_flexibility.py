"""Tests for AutoGen adapter flexibility features: UUID stripping, framework
agent detection, and configurable answer patterns."""

import tempfile
from pathlib import Path

from agent_eval.adapters.autogen.adapter import (
    AUTOGEN_FRAMEWORK_AGENTS,
    AutoGenAdapter,
    strip_uuid_suffix,
)
from agent_eval.core.models import StepKind


class TestStripUuidSuffix:
    def test_strips_full_uuid_suffix(self):
        name = "FinanceExpert_a4bc88a9-3fcd-4dfd-9713-184711810d15/a4bc88a9-3fcd-4dfd-9713-184711810d15"
        assert strip_uuid_suffix(name) == "FinanceExpert"

    def test_strips_single_uuid(self):
        name = "Agent1_a4bc88a9-3fcd-4dfd-9713-184711810d15"
        assert strip_uuid_suffix(name) == "Agent1"

    def test_no_uuid_unchanged(self):
        assert strip_uuid_suffix("FinanceExpert") == "FinanceExpert"

    def test_empty_string(self):
        assert strip_uuid_suffix("") == ""

    def test_framework_agent_with_uuid(self):
        name = "SelectorGroupChatManager_aae2b3b0-6cb6-42c3-9bfc-a1ebb6c4cf94/aae2b3b0-6cb6-42c3-9bfc-a1ebb6c4cf94"
        assert strip_uuid_suffix(name) == "SelectorGroupChatManager"


class TestFrameworkAgentDetection:
    def test_known_framework_agents(self):
        adapter = AutoGenAdapter()
        for name in AUTOGEN_FRAMEWORK_AGENTS:
            assert adapter._is_framework_agent(name)

    def test_framework_agent_with_uuid(self):
        adapter = AutoGenAdapter()
        name = "SelectorGroupChatManager_abc12345-1234-1234-1234-123456789abc/abc12345-1234-1234-1234-123456789abc"
        assert adapter._is_framework_agent(name)

    def test_orchestrator_name_is_framework(self):
        adapter = AutoGenAdapter(orchestrator_name="SalesNegotiator")
        assert adapter._is_framework_agent("SalesNegotiator")

    def test_worker_agent_not_framework(self):
        adapter = AutoGenAdapter()
        assert not adapter._is_framework_agent("FinanceExpert")

    def test_custom_framework_agents(self):
        adapter = AutoGenAdapter(
            framework_agents=frozenset({"MyCustomOrchestrator"})
        )
        assert adapter._is_framework_agent("MyCustomOrchestrator")
        # Default ones are no longer in the set
        assert not adapter._is_framework_agent("SelectorGroupChatManager")


class TestResolveAgentNameWithUuids:
    def test_known_agent_still_matched_by_substring(self):
        adapter = AutoGenAdapter(agent_names=["FinanceExpert"])
        result = adapter._resolve_agent_name(
            "FinanceExpert_a4bc88a9-3fcd-4dfd-9713-184711810d15/a4bc88a9-3fcd-4dfd-9713-184711810d15"
        )
        assert result == "FinanceExpert"

    def test_unknown_agent_uuid_stripped(self):
        adapter = AutoGenAdapter(agent_names=[])
        result = adapter._resolve_agent_name(
            "SomeAgent_a4bc88a9-3fcd-4dfd-9713-184711810d15/a4bc88a9-3fcd-4dfd-9713-184711810d15"
        )
        assert result == "SomeAgent"

    def test_strip_uuids_disabled(self):
        adapter = AutoGenAdapter(agent_names=[], strip_uuids=False)
        raw = "SomeAgent_a4bc88a9-3fcd-4dfd-9713-184711810d15"
        result = adapter._resolve_agent_name(raw)
        assert result == raw  # unchanged


class TestFrameworkAgentTagging:
    def test_framework_steps_tagged(self):
        adapter = AutoGenAdapter(agent_names=[], orchestrator_name="SalesNeg")
        events = [
            {"type": "Message", "sender": "SelectorGroupChatManager_abc12345-1234-1234-1234-123456789abc", "content": ""},
            {"type": "Message", "sender": "FinanceExpert_abc12345-1234-1234-1234-123456789abc", "content": "Analysis"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "event.txt"
            import json
            log_file.write_text("\n".join(json.dumps(e) for e in events))
            episode = adapter.load_episode(str(log_file))

        framework_steps = [
            s for s in episode.steps if s.metadata.get("framework_agent")
        ]
        worker_steps = [
            s for s in episode.steps if not s.metadata.get("framework_agent")
        ]
        assert len(framework_steps) == 1
        assert len(worker_steps) == 1
        assert worker_steps[0].agent_name == "FinanceExpert"


class TestConfigurableAnswerPatterns:
    def test_default_answer_tag(self):
        adapter = AutoGenAdapter()
        content = "Some log...\n<ANSWER>: The revenue is 283M </ANSWER>\nMore log"
        result = adapter._extract_final_answer(content)
        assert result == "The revenue is 283M"

    def test_final_answer_prefix(self):
        adapter = AutoGenAdapter()
        content = "Log data...\nFINAL ANSWER: The result is 42\n\nExtra stuff"
        result = adapter._extract_final_answer(content)
        assert result == "The result is 42"

    def test_custom_pattern(self):
        adapter = AutoGenAdapter(answer_patterns=["## Response:"])
        content = "Log...\n## Response: Here is my answer\n\nMore stuff"
        result = adapter._extract_final_answer(content)
        assert result == "Here is my answer"

    def test_no_answer_found(self):
        adapter = AutoGenAdapter()
        content = "Just some random log content with no markers"
        result = adapter._extract_final_answer(content)
        assert result is None

    def test_first_pattern_wins(self):
        adapter = AutoGenAdapter(
            answer_patterns=["FIRST:", "SECOND:"]
        )
        content = "SECOND: wrong\nFIRST: correct\n\n"
        result = adapter._extract_final_answer(content)
        assert result == "correct"
