"""Tests for janus init and TOML config loading."""
from __future__ import annotations

from janus.cli.init import TEMPLATE, run_init
from janus.config import JanusConfig


def test_init_creates_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_init(non_interactive=True)
    assert result.exists()
    assert result.name == "janus.toml"
    content = result.read_text()
    assert "[janus]" in content
    assert "[risk]" in content
    assert "[circuit_breaker]" in content


def test_init_produces_valid_toml(tmp_path, monkeypatch):
    import tomllib

    monkeypatch.chdir(tmp_path)
    run_init(non_interactive=True)
    toml_path = tmp_path / "janus.toml"
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    assert "janus" in data or "risk" in data


def test_from_toml_roundtrip(tmp_path):
    toml_path = tmp_path / "janus.toml"
    toml_path.write_text(TEMPLATE)
    config = JanusConfig.from_toml(toml_path)
    assert config.risk.lock_threshold == 80.0
    assert config.circuit_breaker.failure_threshold == 5
    assert config.drift.threshold == 0.6
    assert config.log_level == "INFO"


def test_from_toml_custom_values(tmp_path):
    toml_path = tmp_path / "janus.toml"
    toml_path.write_text("""\
[janus]
log_level = "DEBUG"

[risk]
lock_threshold = 90.0

[circuit_breaker]
failure_threshold = 10
""")
    config = JanusConfig.from_toml(toml_path)
    assert config.log_level == "DEBUG"
    assert config.risk.lock_threshold == 90.0
    assert config.circuit_breaker.failure_threshold == 10


def test_config_exporters_defaults():
    config = JanusConfig()
    assert config.exporters.webhook_url == ""
    assert config.exporters.json_log_path == ""
    assert config.exporters.prometheus_enabled is False
    assert config.exporters.otel_enabled is False
    assert config.exporters.otel_service_name == "janus"


def test_config_license_key_default():
    config = JanusConfig()
    assert config.license_key == ""


def test_from_toml_with_exporters(tmp_path):
    toml_path = tmp_path / "janus.toml"
    toml_path.write_text("""\
[janus]
log_level = "INFO"
license_key = "sk-janus-test-key"

[exporters]
webhook_url = "https://example.com/hook"
json_log_path = "/tmp/janus.jsonl"
prometheus_enabled = true
otel_enabled = true
otel_service_name = "my-janus"
""")
    config = JanusConfig.from_toml(toml_path)
    assert config.license_key == "sk-janus-test-key"
    assert config.exporters.webhook_url == "https://example.com/hook"
    assert config.exporters.json_log_path == "/tmp/janus.jsonl"
    assert config.exporters.prometheus_enabled is True
    assert config.exporters.otel_enabled is True
    assert config.exporters.otel_service_name == "my-janus"


def test_init_template_contains_exporters_section(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_init(non_interactive=True)
    content = result.read_text()
    assert "exporters" in content
