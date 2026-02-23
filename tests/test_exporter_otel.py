"""Tests for the OpenTelemetry exporter."""
from __future__ import annotations

import pytest

from sentinel.core.decision import SecurityVerdict, Verdict
from sentinel.exporters.otel import OtelExporter


async def test_otel_span_attributes() -> None:
    exporter = OtelExporter(service_name="sentinel-test")
    verdict = SecurityVerdict(
        verdict=Verdict.BLOCK, risk_score=85.0, risk_delta=50.0,
    )

    # Should not raise even without a real OTEL collector
    exporter.record(verdict, tool_name="api_call", agent_id="agent-1")


async def test_otel_builds_attributes() -> None:
    exporter = OtelExporter(service_name="sentinel-test")
    verdict = SecurityVerdict(
        verdict=Verdict.ALLOW, risk_score=5.0, risk_delta=5.0,
    )

    attrs = exporter.build_attributes(verdict, tool_name="read_file", agent_id="agent-1")
    assert attrs["sentinel.verdict"] == "allow"
    assert attrs["sentinel.risk_score"] == 5.0
    assert attrs["sentinel.tool_name"] == "read_file"
