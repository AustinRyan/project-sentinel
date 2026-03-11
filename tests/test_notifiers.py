"""Tests for notification dispatchers and HMAC webhook signing."""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from janus.config import (
    ExporterConfig,
    NotificationConfig,
    SlackNotificationConfig,
    TelegramNotificationConfig,
)
from janus.core.decision import SecurityVerdict, Verdict
from janus.exporters.coordinator import ExporterCoordinator
from janus.exporters.notifiers import (
    EmailNotifier,
    SlackNotifier,
    TelegramNotifier,
    should_notify,
)
from janus.exporters.webhook import WebhookExporter


def _make_verdict(verdict_val: str = "block", risk: float = 85.0) -> SecurityVerdict:
    return SecurityVerdict(
        verdict=Verdict(verdict_val),
        risk_score=risk,
        risk_delta=20.0,
        reasons=["dangerous payload"],
        trace_id="t-1",
        timestamp=datetime.now(UTC),
    )


# ── should_notify severity filter ──────────────────────────────────

def test_should_notify_block_meets_block():
    assert should_notify("block", "block") is True


def test_should_notify_allow_below_block():
    assert should_notify("allow", "block") is False


def test_should_notify_lock_exceeds_block():
    assert should_notify("lock", "block") is True


def test_should_notify_sandbox_meets_sandbox():
    assert should_notify("sandbox", "sandbox") is True


def test_should_notify_allow_below_sandbox():
    assert should_notify("allow", "sandbox") is False


def test_should_notify_block_exceeds_sandbox():
    assert should_notify("block", "sandbox") is True


# ── Slack Notifier ──────────────────────────────────────────────────

async def test_slack_notifier_sends_on_block():
    notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test", min_verdict="block")
    verdict = _make_verdict("block")

    mock_resp = MagicMock(status_code=200)
    with patch("janus.exporters.notifiers.httpx.AsyncClient") as MockClient:
        client_instance = AsyncMock()
        client_instance.post = AsyncMock(return_value=mock_resp)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await notifier.notify(verdict, tool_name="execute_code", agent_id="a1")
        assert result is True
        client_instance.post.assert_called_once()
        call_kwargs = client_instance.post.call_args
        assert "execute_code" in call_kwargs.kwargs.get("json", call_kwargs[1].get("json", {})).get("text", "")


async def test_slack_notifier_skips_allow():
    notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test", min_verdict="block")
    verdict = _make_verdict("allow", risk=5.0)
    result = await notifier.notify(verdict)
    assert result is False


# ── Telegram Notifier ───────────────────────────────────────────────

async def test_telegram_notifier_sends_on_block():
    notifier = TelegramNotifier(bot_token="123:ABC", chat_id="456", min_verdict="block")
    verdict = _make_verdict("block")

    mock_resp = MagicMock(status_code=200)
    with patch("janus.exporters.notifiers.httpx.AsyncClient") as MockClient:
        client_instance = AsyncMock()
        client_instance.post = AsyncMock(return_value=mock_resp)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await notifier.notify(verdict, tool_name="write_file")
        assert result is True
        call_args = client_instance.post.call_args
        assert "api.telegram.org/bot123:ABC" in call_args[0][0]


async def test_telegram_notifier_skips_allow():
    notifier = TelegramNotifier(bot_token="123:ABC", chat_id="456", min_verdict="block")
    verdict = _make_verdict("allow", risk=2.0)
    result = await notifier.notify(verdict)
    assert result is False


# ── Email Notifier ──────────────────────────────────────────────────

async def test_email_notifier_sends_on_block():
    notifier = EmailNotifier(
        smtp_host="smtp.example.com",
        from_addr="alerts@example.com",
        to_addrs=["admin@example.com"],
        min_verdict="block",
    )
    verdict = _make_verdict("block")

    with patch.object(notifier, "_send_sync") as mock_send:
        result = await notifier.notify(verdict, tool_name="execute_code")
        assert result is True
        mock_send.assert_called_once()


async def test_email_notifier_skips_allow():
    notifier = EmailNotifier(
        smtp_host="smtp.example.com",
        from_addr="a@b.com",
        to_addrs=["c@d.com"],
        min_verdict="block",
    )
    verdict = _make_verdict("allow")
    result = await notifier.notify(verdict)
    assert result is False


# ── HMAC Webhook Signing ────────────────────────────────────────────

async def test_webhook_hmac_signature():
    secret = "test-signing-secret"
    exporter = WebhookExporter(
        url="https://hooks.example.com/test",
        signing_secret=secret,
    )
    verdict = _make_verdict("block")

    captured_headers = {}

    async def mock_post(url, content=None, headers=None, **kwargs):
        captured_headers.update(headers or {})
        resp = MagicMock(status_code=200)
        return resp

    with patch("janus.exporters.webhook.httpx.AsyncClient") as MockClient:
        client_instance = AsyncMock()
        client_instance.post = mock_post
        MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await exporter.send(verdict, tool_name="test")
        assert result is True

    # Verify headers present
    assert "X-Janus-Timestamp" in captured_headers
    assert "X-Janus-Signature" in captured_headers

    # Verify signature is correct
    ts = captured_headers["X-Janus-Timestamp"]
    sig = captured_headers["X-Janus-Signature"]
    payload = exporter.build_payload(verdict, tool_name="test")
    body = json.dumps(payload, default=str)
    expected = hmac.new(
        secret.encode(), f"{ts}.{body}".encode(), hashlib.sha256
    ).hexdigest()
    assert sig == expected


async def test_webhook_no_signature_without_secret():
    exporter = WebhookExporter(url="https://hooks.example.com/test")
    verdict = _make_verdict("allow", risk=5.0)

    captured_headers = {}

    async def mock_post(url, content=None, headers=None, **kwargs):
        captured_headers.update(headers or {})
        return MagicMock(status_code=200)

    with patch("janus.exporters.webhook.httpx.AsyncClient") as MockClient:
        client_instance = AsyncMock()
        client_instance.post = mock_post
        MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        await exporter.send(verdict)

    assert "X-Janus-Signature" not in captured_headers
    assert "X-Janus-Timestamp" not in captured_headers


# ── Coordinator dispatches notifiers ────────────────────────────────

async def test_coordinator_creates_notifiers():
    config = ExporterConfig(
        notifications=NotificationConfig(
            slack=SlackNotificationConfig(webhook_url="https://hooks.slack.com/test"),
            telegram=TelegramNotificationConfig(bot_token="123:ABC", chat_id="456"),
        )
    )
    coord = ExporterCoordinator(config)
    assert coord.enabled_count == 2  # slack + telegram


async def test_coordinator_no_notifiers_by_default():
    config = ExporterConfig()
    coord = ExporterCoordinator(config)
    assert coord.enabled_count == 0
