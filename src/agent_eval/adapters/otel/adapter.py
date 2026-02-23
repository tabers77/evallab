"""OTel GenAI Trace Adapter: converts OTLP JSON trace exports into canonical Episodes.

Supports three input formats:
1. **OTLP JSON** — ``{"resourceSpans": [...]}`` (standard OTel export)
2. **OTLP JSONL** — one ``{"resourceSpans": [...]}`` per line (file exporter)
3. **Flat spans** — ``{"spans": [...]}`` (simplified / test format)

Zero dependencies — pure JSON parsing.  No ``opentelemetry`` package required.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_eval.adapters.otel.span_mapping import (
    get_attr,
    span_to_step,
)
from agent_eval.core.exceptions import AdapterError
from agent_eval.core.models import Episode, Step


class OTelTraceAdapter:
    """Convert OpenTelemetry trace exports into canonical Episodes.

    Parameters
    ----------
    default_agent_name
        Agent name to assign when spans lack ``gen_ai.agent.name``.
    """

    def __init__(self, default_agent_name: str = "agent") -> None:
        self.default_agent_name = default_agent_name

    @property
    def framework_name(self) -> str:
        return "opentelemetry"

    # ── Public API ───────────────────────────────────────────────────

    def load_episode(self, source: str, **kwargs: Any) -> Episode:
        """Load a single Episode from an OTLP JSON file.

        Parameters
        ----------
        source
            Path to a ``.json`` or ``.jsonl`` file containing an OTLP
            trace export.
        """
        trace = self._load_trace(source)
        return self._trace_to_episode(trace, source_path=source)

    def load_episodes(self, source: str, **kwargs: Any) -> list[Episode]:
        """Load Episodes from a directory of OTLP JSON files.

        Finds all ``.json`` and ``.jsonl`` files under *source*
        (recursively) and converts each into an Episode.
        """
        base = Path(source)
        if not base.is_dir():
            return [self.load_episode(source)]

        episodes: list[Episode] = []
        for pattern in ("*.json", "*.jsonl"):
            for trace_file in sorted(base.rglob(pattern)):
                try:
                    ep = self.load_episode(str(trace_file))
                    if ep.steps:
                        episodes.append(ep)
                except Exception:
                    continue
        return episodes

    def load_from_dict(self, trace: dict[str, Any]) -> Episode:
        """Load an Episode directly from a parsed trace dictionary."""
        return self._trace_to_episode(trace, source_path="<dict>")

    # ── Internal ─────────────────────────────────────────────────────

    def _load_trace(self, source: str) -> dict[str, Any]:
        p = Path(source)
        if not p.is_file():
            raise AdapterError(f"Trace file not found: {source}")
        try:
            content = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise AdapterError(f"Cannot read {source}: {exc}") from exc

        # Try JSONL first (one JSON object per line, merge all)
        if source.endswith(".jsonl") or "\n{" in content:
            return self._parse_jsonl(content, source)

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"Invalid JSON in {source}: {exc}") from exc

    def _parse_jsonl(self, content: str, source: str) -> dict[str, Any]:
        """Parse JSONL content, merging all resourceSpans."""
        merged_resource_spans: list[dict] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                merged_resource_spans.extend(obj.get("resourceSpans", []))
            except json.JSONDecodeError:
                continue
        if not merged_resource_spans:
            raise AdapterError(f"No valid OTLP objects found in {source}")
        return {"resourceSpans": merged_resource_spans}

    def _trace_to_episode(
        self, trace: dict[str, Any], source_path: str
    ) -> Episode:
        flat_spans = self._extract_flat_spans(trace)
        if not flat_spans:
            raise AdapterError(
                "Trace has no spans (expected 'resourceSpans' or 'spans' key)"
            )

        # Build parent→children index
        span_by_id: dict[str, dict] = {}
        for span in flat_spans:
            span_by_id[span.get("spanId", "")] = span

        # Sort by startTimeUnixNano for chronological ordering
        def _sort_key(s: dict) -> int:
            try:
                return int(s.get("startTimeUnixNano", 0))
            except (ValueError, TypeError):
                return 0

        flat_spans.sort(key=_sort_key)

        # Convert spans to steps (depth-first via parent lookup)
        steps: list[Step] = []
        for span in flat_spans:
            parent_id = span.get("parentSpanId", "")
            parent_span = span_by_id.get(parent_id) if parent_id else None
            step = span_to_step(span, parent_span, self.default_agent_name)
            steps.append(step)

        # Extract trace-level metadata
        trace_id = self._extract_trace_id(flat_spans)
        service_name = self._extract_service_name(trace)
        scope_name = self._extract_scope_name(trace)

        # Timestamps
        started_at = self._extract_started_at(flat_spans)
        ended_at = self._extract_ended_at(flat_spans)

        return Episode(
            episode_id=trace_id or str(uuid.uuid4()),
            steps=steps,
            source_framework="opentelemetry",
            task_description=None,
            final_answer=None,
            started_at=started_at,
            ended_at=ended_at,
            metadata={
                "source_path": source_path,
                "trace_id": trace_id,
                "service_name": service_name,
                "scope_name": scope_name,
            },
        )

    def _extract_flat_spans(self, trace: dict[str, Any]) -> list[dict]:
        """Flatten all spans from any supported input format."""
        # Standard OTLP: resourceSpans → scopeSpans → spans
        resource_spans = trace.get("resourceSpans", [])
        if resource_spans:
            flat: list[dict] = []
            for rs in resource_spans:
                for ss in rs.get("scopeSpans", []):
                    flat.extend(ss.get("spans", []))
            return flat

        # Simplified format: flat spans list
        if "spans" in trace:
            return list(trace["spans"])

        return []

    @staticmethod
    def _extract_trace_id(spans: list[dict]) -> str | None:
        for span in spans:
            tid = span.get("traceId")
            if tid:
                return tid
        return None

    @staticmethod
    def _extract_service_name(trace: dict[str, Any]) -> str | None:
        for rs in trace.get("resourceSpans", []):
            resource = rs.get("resource", {})
            for attr in resource.get("attributes", []):
                if attr.get("key") == "service.name":
                    val = attr.get("value", {})
                    return val.get("stringValue", str(val))
        return None

    @staticmethod
    def _extract_scope_name(trace: dict[str, Any]) -> str | None:
        for rs in trace.get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                scope = ss.get("scope", {})
                name = scope.get("name")
                if name:
                    return name
        return None

    @staticmethod
    def _nano_to_datetime(nano: str | int | None) -> datetime | None:
        if nano is None:
            return None
        try:
            ts_ns = int(nano)
            return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            return None

    def _extract_started_at(self, spans: list[dict]) -> datetime | None:
        for span in spans:
            dt = self._nano_to_datetime(span.get("startTimeUnixNano"))
            if dt:
                return dt
        return None

    def _extract_ended_at(self, spans: list[dict]) -> datetime | None:
        for span in reversed(spans):
            dt = self._nano_to_datetime(span.get("endTimeUnixNano"))
            if dt:
                return dt
        return None
