"""Tests for the webhook exporter."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from sentinel.core.decision import SecurityVerdict, Verdict
from sentinel.exporters.webhook import WebhookExporter


async def test_webhook_sends_verdict() -> None:
    exporter = WebhookExporter(url="https://hooks.example.com/sentinel")

    mock_response = AsyncMock()
    mock_response.status_code = 200

    with patch("sentinel.exporters.webhook.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        verdict = SecurityVerdict(
            verdict=Verdict.BLOCK, risk_score=85.0, risk_delta=50.0,
            reasons=["Pattern match detected"],
        )
        await exporter.send(verdict, tool_name="api_call", agent_id="agent-1")
        mock_client.post.assert_called_once()


async def test_webhook_payload_structure() -> None:
    exporter = WebhookExporter(url="https://hooks.example.com/sentinel")
    verdict = SecurityVerdict(
        verdict=Verdict.ALLOW, risk_score=5.0, risk_delta=5.0,
    )
    payload = exporter.build_payload(verdict, tool_name="read_file", agent_id="agent-1")
    assert payload["verdict"] == "allow"
    assert payload["risk_score"] == 5.0
    assert payload["tool_name"] == "read_file"
    assert "timestamp" in payload
