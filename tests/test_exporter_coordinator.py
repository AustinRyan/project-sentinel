"""Tests for ExporterCoordinator."""
from __future__ import annotations

from datetime import datetime

from janus.config import ExporterConfig
from janus.core.decision import SecurityVerdict, Verdict
from janus.exporters.coordinator import ExporterCoordinator


def _make_verdict(**overrides) -> SecurityVerdict:
    defaults = {
        "verdict": Verdict.ALLOW,
        "risk_score": 10.0,
        "risk_delta": 5.0,
        "reasons": ["test"],
        "timestamp": datetime.now(),
        "trace_id": "trace-001",
    }
    defaults.update(overrides)
    return SecurityVerdict(**defaults)


def test_coordinator_no_exporters_enabled():
    config = ExporterConfig()
    coordinator = ExporterCoordinator(config)
    assert coordinator.enabled_count == 0


def test_coordinator_webhook_enabled():
    config = ExporterConfig(webhook_url="https://example.com/hook")
    coordinator = ExporterCoordinator(config)
    assert coordinator.enabled_count == 1


def test_coordinator_json_log_enabled():
    config = ExporterConfig(json_log_path="/tmp/test.jsonl")
    coordinator = ExporterCoordinator(config)
    assert coordinator.enabled_count == 1


def test_coordinator_prometheus_enabled():
    config = ExporterConfig(prometheus_enabled=True)
    coordinator = ExporterCoordinator(config)
    assert coordinator.enabled_count == 1


def test_coordinator_otel_enabled():
    config = ExporterConfig(otel_enabled=True)
    coordinator = ExporterCoordinator(config)
    assert coordinator.enabled_count == 1


def test_coordinator_all_enabled():
    config = ExporterConfig(
        webhook_url="https://example.com/hook",
        json_log_path="-",
        prometheus_enabled=True,
        otel_enabled=True,
    )
    coordinator = ExporterCoordinator(config)
    assert coordinator.enabled_count == 4


async def test_coordinator_export_no_exporters():
    config = ExporterConfig()
    coordinator = ExporterCoordinator(config)
    verdict = _make_verdict()
    # Should not raise
    await coordinator.export(verdict, tool_name="test_tool", agent_id="agent-1", session_id="s-1")


async def test_coordinator_export_with_json_log(tmp_path):
    log_path = str(tmp_path / "verdicts.jsonl")
    config = ExporterConfig(json_log_path=log_path)
    coordinator = ExporterCoordinator(config)
    verdict = _make_verdict()

    await coordinator.export(verdict, tool_name="read_file", agent_id="agent-1", session_id="s-1")

    import json
    from pathlib import Path

    content = Path(log_path).read_text().strip()
    data = json.loads(content)
    assert data["verdict"] == "allow"
    assert data["tool_name"] == "read_file"
    assert data["agent_id"] == "agent-1"


async def test_coordinator_export_with_prometheus():
    config = ExporterConfig(prometheus_enabled=True)
    coordinator = ExporterCoordinator(config)
    verdict = _make_verdict()

    await coordinator.export(verdict, tool_name="test_tool", session_id="s-1")
    # Verify internal tracking (always available even without prometheus_client)
    assert coordinator._prometheus.get_verdict_count("allow") == 1
