"""Convenience logger that captures AutoGen events to a JSONL file.

Usage::

    from agent_eval.adapters.autogen import attach_logger

    handle = attach_logger("logs/events.jsonl")
    # ... run AutoGen team ...
    handle.detach()

The resulting file is directly loadable by :class:`AutoGenAdapter`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------


def _normalize(record_msg: Any) -> dict | None:
    """Turn a log record message into a dict, or *None* if unusable.

    Handles:
    - ``dict`` — returned as-is (must contain ``"type"``).
    - Pydantic-like objects — calls ``model_dump()`` or ``dict()``.
    - ``str`` — attempts ``json.loads``; must produce a dict with ``"type"``.
    - Anything else — returns *None*.
    """
    obj: Any = record_msg

    if isinstance(obj, dict):
        return obj if "type" in obj else None

    # Pydantic v2
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
        if isinstance(obj, dict) and "type" in obj:
            return obj
        return None

    # Pydantic v1
    if hasattr(obj, "dict") and callable(obj.dict):
        obj = obj.dict()
        if isinstance(obj, dict) and "type" in obj:
            return obj
        return None

    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict) and "type" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    return None


# ---------------------------------------------------------------------------
# Custom logging handler
# ---------------------------------------------------------------------------


class _JsonlEventHandler(logging.Handler):
    """Write AutoGen event records as JSON lines to *stream*."""

    def __init__(self, stream: Any, level: int = logging.DEBUG) -> None:
        super().__init__(level)
        self._stream = stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = _normalize(record.msg)
            if event is None:
                return
            line = json.dumps(event, default=str)
            self._stream.write(line + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Handle returned to caller
# ---------------------------------------------------------------------------


@dataclass
class LoggerHandle:
    """Opaque handle returned by :func:`attach_logger`.

    Call :meth:`detach` to stop capturing events and close the file.
    """

    path: Path
    _handler: _JsonlEventHandler = field(repr=False)
    _logger: logging.Logger = field(repr=False)
    _stream: Any = field(repr=False)

    def detach(self) -> None:
        """Remove the handler from the logger and close the file."""
        self._logger.removeHandler(self._handler)
        self._stream.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attach_logger(
    path: str | Path,
    *,
    level: int = logging.DEBUG,
    encoding: str = "utf-8",
) -> LoggerHandle:
    """Attach a JSONL event handler to AutoGen's event logger.

    Parameters
    ----------
    path
        Destination file.  Parent directories are created automatically.
        The file is opened in **append** mode so multiple sessions can
        safely write to the same file.
    level
        Minimum log level to capture (default ``DEBUG``).
    encoding
        File encoding (default ``utf-8``).

    Returns
    -------
    LoggerHandle
        Call ``.detach()`` when done to stop capturing and close the file.

    Raises
    ------
    ImportError
        If ``autogen_core`` is not installed.
    """
    try:
        from autogen_core import EVENT_LOGGER_NAME  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "autogen_core is required for attach_logger. "
            "Install it with: pip install autogen-core"
        ) from exc

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    stream = open(dest, mode="a", encoding=encoding)  # noqa: SIM115
    handler = _JsonlEventHandler(stream, level=level)

    logger = logging.getLogger(EVENT_LOGGER_NAME)
    logger.addHandler(handler)

    return LoggerHandle(
        path=dest,
        _handler=handler,
        _logger=logger,
        _stream=stream,
    )
