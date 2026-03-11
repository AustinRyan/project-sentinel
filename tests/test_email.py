"""Tests for janus.email — Resend license key delivery."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from janus.email import _build_email_html, send_license_email


def test_send_license_email_returns_false_without_api_key(monkeypatch) -> None:
    """send_license_email returns False when RESEND_API_KEY is not set."""
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    result = send_license_email(to="test@example.com", license_key="sk-janus-test-abc")
    assert result is False


def test_send_license_email_returns_false_without_resend_package(monkeypatch) -> None:
    """send_license_email returns False when resend is not installed."""
    monkeypatch.setenv("RESEND_API_KEY", "re_test_123")

    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "resend":
            raise ImportError("No module named 'resend'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        result = send_license_email(to="test@example.com", license_key="sk-janus-test-abc")
        assert result is False


def test_send_license_email_returns_false_on_api_error(monkeypatch) -> None:
    """send_license_email returns False (never raises) on Resend API error."""
    monkeypatch.setenv("RESEND_API_KEY", "re_test_123")

    mock_resend = MagicMock()
    mock_resend.Emails.send.side_effect = RuntimeError("API error")

    with patch.dict("sys.modules", {"resend": mock_resend}):
        result = send_license_email(to="test@example.com", license_key="sk-janus-test-abc")
        assert result is False


def test_send_license_email_success(monkeypatch) -> None:
    """send_license_email returns True and calls Resend correctly on success."""
    monkeypatch.setenv("RESEND_API_KEY", "re_test_123")

    mock_resend = MagicMock()
    mock_resend.Emails.send.return_value = {"id": "email_123"}

    with patch.dict("sys.modules", {"resend": mock_resend}):
        result = send_license_email(
            to="buyer@example.com",
            license_key="sk-janus-payload-sig",
            tier="pro",
        )
        assert result is True

        # Verify the call was made with correct parameters
        call_args = mock_resend.Emails.send.call_args[0][0]
        assert call_args["to"] == ["buyer@example.com"]
        assert "Janus Pro" in call_args["subject"]
        assert "sk-janus-payload-sig" in call_args["html"]
        assert call_args["from"] == "Janus Security <noreply@janus-security.dev>"


def test_build_email_html_contains_key() -> None:
    """Email HTML contains the license key and activation instructions."""
    html = _build_email_html("sk-janus-test-key-abc123", "pro")
    assert "sk-janus-test-key-abc123" in html
    assert "janus.toml" in html
    assert "/api/license/activate" in html
    assert "Janus Pro" in html


def test_build_email_html_tier_capitalization() -> None:
    """Email HTML capitalizes the tier name correctly."""
    html = _build_email_html("sk-janus-x-y", "enterprise")
    assert "Janus Enterprise" in html
