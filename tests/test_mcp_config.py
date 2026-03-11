"""Tests for MCP proxy configuration."""
from __future__ import annotations

import os
import tempfile

from janus.mcp.config import (
    AgentConfig,
    ProxyConfig,
    ProxyTransportConfig,
    TransportType,
    UpstreamServerConfig,
)


def test_default_config() -> None:
    config = ProxyConfig()
    assert config.server_name == "janus-proxy"
    assert config.database_path == ":memory:"
    assert config.agent.agent_id == "mcp-proxy-agent"
    assert config.transport.type == TransportType.STDIO
    assert config.upstream_servers == []


def test_upstream_server_defaults() -> None:
    cfg = UpstreamServerConfig(name="test")
    assert cfg.transport == TransportType.STDIO
    assert cfg.command == ""
    assert cfg.namespace == ""
    assert cfg.timeout == 30.0


def test_env_resolution() -> None:
    os.environ["TEST_TOKEN_XYZ"] = "secret123"
    try:
        cfg = UpstreamServerConfig(
            name="test",
            env={"TOKEN": "${TEST_TOKEN_XYZ}", "PLAIN": "hello"},
        )
        resolved = cfg.resolve_env()
        assert resolved["TOKEN"] == "secret123"
        assert resolved["PLAIN"] == "hello"
    finally:
        del os.environ["TEST_TOKEN_XYZ"]


def test_env_resolution_missing_var() -> None:
    cfg = UpstreamServerConfig(
        name="test",
        env={"TOKEN": "${NONEXISTENT_VAR_ABC}"},
    )
    resolved = cfg.resolve_env()
    assert resolved["TOKEN"] == ""


def test_namespace_config() -> None:
    cfg = UpstreamServerConfig(
        name="github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        namespace="gh",
    )
    assert cfg.namespace == "gh"
    assert cfg.args == ["-y", "@modelcontextprotocol/server-github"]


def test_from_toml() -> None:
    toml_content = b"""
server_name = "my-proxy"
log_level = "DEBUG"

[agent]
agent_id = "claude-desktop"
role = "admin"
permissions = ["*"]

[[upstream_servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

[[upstream_servers]]
name = "github"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
namespace = "gh"
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        config = ProxyConfig.from_toml(f.name)

    assert config.server_name == "my-proxy"
    assert config.log_level == "DEBUG"
    assert config.agent.agent_id == "claude-desktop"
    assert config.agent.role == "admin"
    assert len(config.upstream_servers) == 2
    assert config.upstream_servers[0].name == "filesystem"
    assert config.upstream_servers[1].namespace == "gh"
    os.unlink(f.name)


def test_agent_config_defaults() -> None:
    cfg = AgentConfig()
    assert cfg.permissions == ["*"]
    assert cfg.original_goal == ""


def test_http_transport_config() -> None:
    cfg = ProxyTransportConfig(type=TransportType.HTTP, port=9000)
    assert cfg.type == TransportType.HTTP
    assert cfg.port == 9000


def test_multiple_upstreams_with_mixed_transport() -> None:
    config = ProxyConfig(
        upstream_servers=[
            UpstreamServerConfig(
                name="fs", transport=TransportType.STDIO, command="echo",
            ),
            UpstreamServerConfig(
                name="api", transport=TransportType.HTTP, url="http://localhost:9000",
            ),
        ]
    )
    assert len(config.upstream_servers) == 2
    assert config.upstream_servers[0].transport == TransportType.STDIO
    assert config.upstream_servers[1].transport == TransportType.HTTP
    assert config.upstream_servers[1].url == "http://localhost:9000"
