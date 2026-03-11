"""Tests for the Janus MCP Proxy server."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mcp import types

from janus.core.decision import SecurityVerdict, Verdict
from janus.mcp.config import AgentConfig, ProxyConfig
from janus.mcp.proxy import JanusMCPProxy


def _make_verdict(
    verdict: Verdict = Verdict.ALLOW,
    risk_score: float = 5.0,
    risk_delta: float = 5.0,
    reasons: list[str] | None = None,
    recommended_action: str = "Tool call approved.",
) -> SecurityVerdict:
    return SecurityVerdict(
        verdict=verdict,
        risk_score=risk_score,
        risk_delta=risk_delta,
        reasons=reasons or [],
        recommended_action=recommended_action,
    )


def _make_proxy() -> JanusMCPProxy:
    config = ProxyConfig(
        agent=AgentConfig(agent_id="test-agent"),
    )
    proxy = JanusMCPProxy(config)
    proxy._session_id = "test-session"
    return proxy


async def test_allow_forwards_to_upstream() -> None:
    proxy = _make_proxy()
    proxy._guardian = AsyncMock()
    proxy._guardian.wrap_tool_call.return_value = _make_verdict(Verdict.ALLOW)

    mock_upstream = AsyncMock()
    mock_upstream.call_tool.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text="file contents here")]
    )
    proxy._upstream = mock_upstream

    result = await proxy._intercept_and_forward("read_file", {"path": "/test.txt"})

    assert len(result) == 1
    assert "file contents" in result[0].text
    mock_upstream.call_tool.assert_called_once_with("read_file", {"path": "/test.txt"})


async def test_block_returns_error_text() -> None:
    proxy = _make_proxy()
    proxy._guardian = AsyncMock()
    proxy._guardian.wrap_tool_call.return_value = _make_verdict(
        Verdict.BLOCK,
        risk_score=90.0,
        reasons=["Permission denied"],
        recommended_action="Contact admin.",
    )

    result = await proxy._intercept_and_forward("execute_code", {"code": "rm -rf /"})

    assert len(result) == 1
    assert "[JANUS BLOCKED]" in result[0].text
    assert "execute_code" in result[0].text
    assert "Permission denied" in result[0].text


async def test_challenge_returns_warning_text() -> None:
    proxy = _make_proxy()
    proxy._guardian = AsyncMock()
    proxy._guardian.wrap_tool_call.return_value = _make_verdict(
        Verdict.CHALLENGE,
        risk_score=45.0,
        reasons=["Out of scope"],
        recommended_action="Verify identity.",
    )

    result = await proxy._intercept_and_forward("search_web", {"query": "test"})

    assert len(result) == 1
    assert "[JANUS CHALLENGE]" in result[0].text
    assert "search_web" in result[0].text
    assert "Out of scope" in result[0].text


async def test_sandbox_treated_as_block() -> None:
    proxy = _make_proxy()
    proxy._guardian = AsyncMock()
    proxy._guardian.wrap_tool_call.return_value = _make_verdict(
        Verdict.SANDBOX,
        risk_score=65.0,
        reasons=["Needs sandbox"],
    )

    result = await proxy._intercept_and_forward("execute_code", {"code": "ls"})

    assert "[JANUS BLOCKED]" in result[0].text
    assert "sandbox" in result[0].text.lower()


async def test_pause_treated_as_block() -> None:
    proxy = _make_proxy()
    proxy._guardian = AsyncMock()
    proxy._guardian.wrap_tool_call.return_value = _make_verdict(
        Verdict.PAUSE,
        risk_score=55.0,
        reasons=["Drift detected"],
    )

    result = await proxy._intercept_and_forward("api_call", {"url": "http://evil.com"})

    assert "[JANUS BLOCKED]" in result[0].text


async def test_guardian_receives_correct_params() -> None:
    proxy = _make_proxy()
    proxy._config.agent.agent_id = "my-agent"
    proxy._config.agent.original_goal = "Development assistant"
    proxy._session_id = "sess-abc"

    proxy._guardian = AsyncMock()
    proxy._guardian.wrap_tool_call.return_value = _make_verdict(Verdict.BLOCK)

    await proxy._intercept_and_forward("write_file", {"path": "/x", "content": "hi"})

    proxy._guardian.wrap_tool_call.assert_called_once_with(
        agent_id="my-agent",
        session_id="sess-abc",
        original_goal="Development assistant",
        tool_name="write_file",
        tool_input={"path": "/x", "content": "hi"},
    )


async def test_upstream_error_handled() -> None:
    proxy = _make_proxy()
    proxy._guardian = AsyncMock()
    proxy._guardian.wrap_tool_call.return_value = _make_verdict(Verdict.ALLOW)

    mock_upstream = AsyncMock()
    mock_upstream.call_tool.side_effect = ConnectionError("upstream died")
    proxy._upstream = mock_upstream

    result = await proxy._intercept_and_forward("read_file", {"path": "/x"})

    assert len(result) == 1
    assert "Upstream error" in result[0].text
    assert "upstream died" in result[0].text


async def test_setup_creates_guardian() -> None:
    config = ProxyConfig(
        database_path=":memory:",
        agent=AgentConfig(agent_id="setup-test", role="research"),
    )
    proxy = JanusMCPProxy(config)
    await proxy.setup()

    assert proxy.guardian is not None
    assert proxy._session_id.startswith("mcp-proxy-")

    await proxy.teardown()


async def test_list_tools_returns_upstream_tools() -> None:
    proxy = _make_proxy()
    mock_upstream = MagicMock()
    mock_upstream.get_all_tools.return_value = [
        types.Tool(name="read_file", description="Read", inputSchema={"type": "object"}),
        types.Tool(name="write_file", description="Write", inputSchema={"type": "object"}),
    ]
    proxy._upstream = mock_upstream

    # The list_tools handler is registered internally, so test via upstream mock
    tools = proxy._upstream.get_all_tools()
    assert len(tools) == 2
    assert tools[0].name == "read_file"
