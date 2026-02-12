"""Tests for adapters.autogen.logger — attach_logger utility.

These tests do NOT require ``autogen_core``; they exercise the handler
and normalisation logic using a plain Python logger.
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
from uuid import UUID

import pytest

from agent_eval.adapters.autogen.logger import (
    LoggerHandle,
    _JsonlEventHandler,
    _normalize,
)

# A unique logger name so tests don't clash with real AutoGen loggers.
_TEST_LOGGER = "agent_eval.test.logger"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _FakePydanticV2:
    """Mimics a Pydantic v2 model with model_dump()."""

    def model_dump(self) -> dict:
        return {"type": "Message", "sender": "Agent1", "content": "hi"}


class _FakePydanticV1:
    """Mimics a Pydantic v1 model with dict()."""

    def dict(self) -> dict:
        return {"type": "ToolCall", "tool": "search", "arguments": "q=1"}


class _FakePydanticNoType:
    """Pydantic-like model whose dump has no 'type' key."""

    def model_dump(self) -> dict:
        return {"sender": "Agent1", "content": "hi"}


# ------------------------------------------------------------------
# _normalize tests
# ------------------------------------------------------------------


class TestNormalize:
    def test_dict_with_type(self):
        d = {"type": "Message", "sender": "A"}
        assert _normalize(d) == d

    def test_dict_without_type_returns_none(self):
        assert _normalize({"sender": "A"}) is None

    def test_pydantic_v2_model_dump(self):
        result = _normalize(_FakePydanticV2())
        assert result == {"type": "Message", "sender": "Agent1", "content": "hi"}

    def test_pydantic_v1_dict(self):
        result = _normalize(_FakePydanticV1())
        assert result == {"type": "ToolCall", "tool": "search", "arguments": "q=1"}

    def test_pydantic_no_type_returns_none(self):
        assert _normalize(_FakePydanticNoType()) is None

    def test_json_string_with_type(self):
        s = json.dumps({"type": "LLMCall", "model": "gpt-4"})
        result = _normalize(s)
        assert result == {"type": "LLMCall", "model": "gpt-4"}

    def test_json_string_without_type_returns_none(self):
        s = json.dumps({"model": "gpt-4"})
        assert _normalize(s) is None

    def test_plain_string_returns_none(self):
        assert _normalize("just a log message") is None

    def test_invalid_json_string_returns_none(self):
        assert _normalize("{bad json") is None

    def test_none_returns_none(self):
        assert _normalize(None) is None

    def test_int_returns_none(self):
        assert _normalize(42) is None

    def test_empty_dict_returns_none(self):
        assert _normalize({}) is None


# ------------------------------------------------------------------
# _JsonlEventHandler tests
# ------------------------------------------------------------------


class TestJsonlEventHandler:
    def _make_record(self, msg):
        record = logging.LogRecord(
            name=_TEST_LOGGER,
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg=msg,
            args=None,
            exc_info=None,
        )
        return record

    def test_writes_valid_jsonl(self):
        buf = io.StringIO()
        handler = _JsonlEventHandler(buf)

        handler.emit(self._make_record({"type": "Message", "sender": "A"}))

        buf.seek(0)
        line = buf.readline().strip()
        parsed = json.loads(line)
        assert parsed == {"type": "Message", "sender": "A"}

    def test_skips_events_without_type(self):
        buf = io.StringIO()
        handler = _JsonlEventHandler(buf)

        handler.emit(self._make_record({"sender": "A"}))

        buf.seek(0)
        assert buf.read() == ""

    def test_handles_non_serializable_fields(self):
        """datetime and UUID should be serialized via default=str."""
        buf = io.StringIO()
        handler = _JsonlEventHandler(buf)
        now = datetime(2025, 1, 15, 10, 30, 0)
        uid = UUID("12345678-1234-5678-1234-567812345678")

        handler.emit(
            self._make_record(
                {"type": "Message", "timestamp": now, "id": uid}
            )
        )

        buf.seek(0)
        parsed = json.loads(buf.readline())
        assert parsed["type"] == "Message"
        assert "2025-01-15" in parsed["timestamp"]
        assert "12345678" in parsed["id"]

    def test_multiple_events_produce_multiple_lines(self):
        buf = io.StringIO()
        handler = _JsonlEventHandler(buf)

        for i in range(3):
            handler.emit(
                self._make_record({"type": "Message", "seq": i})
            )

        buf.seek(0)
        lines = [l for l in buf.readlines() if l.strip()]
        assert len(lines) == 3
        for i, line in enumerate(lines):
            assert json.loads(line)["seq"] == i

    def test_skips_plain_string(self):
        buf = io.StringIO()
        handler = _JsonlEventHandler(buf)
        handler.emit(self._make_record("plain text log"))
        buf.seek(0)
        assert buf.read() == ""


# ------------------------------------------------------------------
# attach_logger integration tests
# ------------------------------------------------------------------


class TestAttachLogger:
    def test_creates_file_and_writes_events(self, tmp_path):
        dest = tmp_path / "events.jsonl"
        logger = logging.getLogger(_TEST_LOGGER)
        logger.setLevel(logging.DEBUG)

        # Manually wire up since we don't have autogen_core
        stream = open(dest, "a", encoding="utf-8")
        handler = _JsonlEventHandler(stream)
        logger.addHandler(handler)
        handle = LoggerHandle(
            path=dest, _handler=handler, _logger=logger, _stream=stream
        )

        logger.debug({"type": "Message", "sender": "Agent1", "content": "hello"})
        logger.debug({"type": "ToolCall", "tool": "search"})

        handle.detach()

        lines = dest.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "Message"
        assert json.loads(lines[1])["type"] == "ToolCall"

    def test_detach_stops_capture(self, tmp_path):
        dest = tmp_path / "events.jsonl"
        logger = logging.getLogger(_TEST_LOGGER + ".detach")
        logger.setLevel(logging.DEBUG)

        stream = open(dest, "a", encoding="utf-8")
        handler = _JsonlEventHandler(stream)
        logger.addHandler(handler)
        handle = LoggerHandle(
            path=dest, _handler=handler, _logger=logger, _stream=stream
        )

        logger.debug({"type": "Message", "content": "before"})
        handle.detach()
        logger.debug({"type": "Message", "content": "after"})

        lines = [l for l in dest.read_text().strip().split("\n") if l]
        assert len(lines) == 1
        assert json.loads(lines[0])["content"] == "before"

    def test_append_mode(self, tmp_path):
        dest = tmp_path / "events.jsonl"
        dest.write_text('{"type": "Message", "content": "existing"}\n')

        logger = logging.getLogger(_TEST_LOGGER + ".append")
        logger.setLevel(logging.DEBUG)
        stream = open(dest, "a", encoding="utf-8")
        handler = _JsonlEventHandler(stream)
        logger.addHandler(handler)
        handle = LoggerHandle(
            path=dest, _handler=handler, _logger=logger, _stream=stream
        )

        logger.debug({"type": "Message", "content": "new"})
        handle.detach()

        lines = [l for l in dest.read_text().strip().split("\n") if l]
        assert len(lines) == 2
        assert json.loads(lines[0])["content"] == "existing"
        assert json.loads(lines[1])["content"] == "new"

    def test_parent_dirs_created(self, tmp_path):
        dest = tmp_path / "a" / "b" / "c" / "events.jsonl"
        logger = logging.getLogger(_TEST_LOGGER + ".dirs")
        logger.setLevel(logging.DEBUG)

        dest.parent.mkdir(parents=True, exist_ok=True)
        stream = open(dest, "a", encoding="utf-8")
        handler = _JsonlEventHandler(stream)
        logger.addHandler(handler)
        handle = LoggerHandle(
            path=dest, _handler=handler, _logger=logger, _stream=stream
        )

        logger.debug({"type": "Message", "content": "deep"})
        handle.detach()

        assert dest.exists()
        lines = [l for l in dest.read_text().strip().split("\n") if l]
        assert len(lines) == 1

    def test_output_parseable_by_adapter(self, tmp_path):
        """The JSONL output should be loadable by AutoGenAdapter."""
        from agent_eval.adapters.autogen.adapter import AutoGenAdapter

        dest = tmp_path / "events.jsonl"
        logger = logging.getLogger(_TEST_LOGGER + ".adapter")
        logger.setLevel(logging.DEBUG)

        stream = open(dest, "a", encoding="utf-8")
        handler = _JsonlEventHandler(stream)
        logger.addHandler(handler)
        handle = LoggerHandle(
            path=dest, _handler=handler, _logger=logger, _stream=stream
        )

        logger.debug(
            {
                "type": "Message",
                "sender": "TestAgent",
                "content": "hello world",
                "timestamp": "2025-01-15T10:30:00Z",
            }
        )
        logger.debug(
            {
                "type": "ToolCall",
                "tool": "calculator",
                "arguments": "2+2",
                "result": "4",
                "timestamp": "2025-01-15T10:30:01Z",
            }
        )
        handle.detach()

        adapter = AutoGenAdapter(agent_names=["TestAgent"])
        episode = adapter.load_episode(str(dest))
        assert len(episode.steps) == 2
        assert episode.steps[0].kind.value == "message"
        assert episode.steps[1].kind.value == "tool_call"

    def test_import_error_when_autogen_missing(self):
        """attach_logger raises ImportError when autogen_core is absent."""
        from agent_eval.adapters.autogen.logger import attach_logger

        with patch.dict("sys.modules", {"autogen_core": None}):
            with pytest.raises(ImportError, match="autogen_core"):
                attach_logger("some/path.jsonl")
