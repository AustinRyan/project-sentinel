"""Tests for the Prometheus metrics exporter."""
from __future__ import annotations

from janus.core.decision import SecurityVerdict, Verdict
from janus.exporters.prometheus import PrometheusExporter


async def test_prometheus_records_verdict() -> None:
    exporter = PrometheusExporter()
    verdict = SecurityVerdict(
        verdict=Verdict.ALLOW, risk_score=5.0, risk_delta=5.0,
    )
    exporter.record(verdict, tool_name="read_file")
    assert exporter.get_verdict_count("allow") >= 1


async def test_prometheus_records_block() -> None:
    exporter = PrometheusExporter()
    verdict = SecurityVerdict(
        verdict=Verdict.BLOCK, risk_score=85.0, risk_delta=50.0,
    )
    exporter.record(verdict, tool_name="api_call")
    assert exporter.get_verdict_count("block") >= 1


async def test_prometheus_tracks_risk() -> None:
    exporter = PrometheusExporter()
    verdict = SecurityVerdict(
        verdict=Verdict.ALLOW, risk_score=42.0, risk_delta=5.0,
    )
    exporter.record(verdict, session_id="sess-1")
    assert exporter.get_risk_score("sess-1") == 42.0
