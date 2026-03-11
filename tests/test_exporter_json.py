"""Tests for the JSON logger exporter."""
from __future__ import annotations

import json
from io import StringIO

from janus.core.decision import SecurityVerdict, Verdict
from janus.exporters.json_logger import JsonLogExporter


async def test_json_log_to_stream() -> None:
    stream = StringIO()
    exporter = JsonLogExporter(stream=stream)

    verdict = SecurityVerdict(
        verdict=Verdict.BLOCK, risk_score=85.0, risk_delta=50.0,
        reasons=["Sleeper pattern matched"],
    )
    exporter.log(verdict, tool_name="api_call", agent_id="agent-1")

    output = stream.getvalue().strip()
    data = json.loads(output)
    assert data["verdict"] == "block"
    assert data["risk_score"] == 85.0
    assert data["tool_name"] == "api_call"


async def test_json_log_is_one_line() -> None:
    stream = StringIO()
    exporter = JsonLogExporter(stream=stream)

    for i in range(3):
        verdict = SecurityVerdict(
            verdict=Verdict.ALLOW, risk_score=float(i), risk_delta=float(i),
        )
        exporter.log(verdict, tool_name=f"tool_{i}")

    lines = stream.getvalue().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        json.loads(line)  # Should not raise
