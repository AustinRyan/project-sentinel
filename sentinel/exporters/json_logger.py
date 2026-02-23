"""Structured JSON log exporter — one JSON line per verdict.

Compatible with Splunk, Elastic, Datadog log ingestion.
"""
from __future__ import annotations

import json
import sys
from typing import IO, Any

from sentinel.core.decision import SecurityVerdict


class JsonLogExporter:
    """Writes SecurityVerdicts as JSON lines to a stream."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream or sys.stdout

    def log(
        self,
        verdict: SecurityVerdict,
        tool_name: str = "",
        agent_id: str = "",
        session_id: str = "",
    ) -> None:
        record: dict[str, Any] = {
            "verdict": verdict.verdict.value,
            "risk_score": verdict.risk_score,
            "risk_delta": verdict.risk_delta,
            "reasons": verdict.reasons,
            "drift_score": verdict.drift_score,
            "itdr_signals": verdict.itdr_signals,
            "trace_id": verdict.trace_id,
            "recommended_action": verdict.recommended_action,
            "timestamp": verdict.timestamp.isoformat(),
            "tool_name": tool_name,
            "agent_id": agent_id,
            "session_id": session_id,
            "source": "sentinel",
        }
        self._stream.write(json.dumps(record, default=str) + "\n")
        self._stream.flush()
