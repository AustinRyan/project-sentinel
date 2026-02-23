"""Tests for the Claude MCP integration adapter."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from sentinel.core.decision import SecurityVerdict, Verdict
from sentinel.integrations.mcp import SentinelMCPServer, MCPToolDefinition


async def test_mcp_server_registers_tools() -> None:
    mock_guardian = AsyncMock()
    server = SentinelMCPServer(
        guardian=mock_guardian,
        agent_id="test-agent",
        session_id="test-session",
    )

    server.add_tool(MCPToolDefinition(
        name="read_file",
        description="Read a file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        handler=AsyncMock(return_value={"content": "file data"}),
    ))

    assert "read_file" in server.tool_names


async def test_mcp_server_allows_tool() -> None:
    mock_guardian = AsyncMock()
    mock_guardian.wrap_tool_call.return_value = SecurityVerdict(
        verdict=Verdict.ALLOW, risk_score=5.0, risk_delta=5.0,
    )

    handler = AsyncMock(return_value={"content": "file data"})
    server = SentinelMCPServer(
        guardian=mock_guardian,
        agent_id="test-agent",
        session_id="test-session",
    )
    server.add_tool(MCPToolDefinition(
        name="read_file",
        description="Read a file",
        input_schema={},
        handler=handler,
    ))

    result = await server.call_tool("read_file", {"path": "/test.txt"})
    assert result == {"content": "file data"}
    handler.assert_called_once()


async def test_mcp_server_blocks_tool() -> None:
    mock_guardian = AsyncMock()
    mock_guardian.wrap_tool_call.return_value = SecurityVerdict(
        verdict=Verdict.BLOCK, risk_score=90.0, risk_delta=50.0,
        recommended_action="Blocked",
    )

    handler = AsyncMock()
    server = SentinelMCPServer(
        guardian=mock_guardian,
        agent_id="test-agent",
        session_id="test-session",
    )
    server.add_tool(MCPToolDefinition(
        name="execute_code",
        description="Execute code",
        input_schema={},
        handler=handler,
    ))

    result = await server.call_tool("execute_code", {"code": "rm -rf /"})
    assert "blocked" in str(result).lower()
    handler.assert_not_called()
