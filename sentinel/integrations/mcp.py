"""Claude MCP (Model Context Protocol) integration adapter for Sentinel.

Acts as a security middleware MCP server that wraps tool definitions.

Usage::

    from sentinel.integrations.mcp import SentinelMCPServer, MCPToolDefinition

    server = SentinelMCPServer(guardian=guardian, agent_id="a-1", session_id="s-1")
    server.add_tool(MCPToolDefinition(
        name="read_file", description="Read a file",
        input_schema={...}, handler=my_read_handler,
    ))
    result = await server.call_tool("read_file", {"path": "/test.txt"})
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sentinel.core.decision import Verdict
from sentinel.core.guardian import Guardian


@dataclass
class MCPToolDefinition:
    """Definition of a tool exposed through the MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]


class SentinelMCPServer:
    """MCP server that wraps tool definitions with Guardian security."""

    def __init__(
        self,
        guardian: Guardian,
        agent_id: str,
        session_id: str,
        original_goal: str = "",
    ) -> None:
        self._guardian = guardian
        self._agent_id = agent_id
        self._session_id = session_id
        self._original_goal = original_goal
        self._tools: dict[str, MCPToolDefinition] = {}

    def add_tool(self, tool: MCPToolDefinition) -> None:
        self._tools[tool.name] = tool

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions in MCP format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        """Execute a tool call with Guardian interception."""
        verdict = await self._guardian.wrap_tool_call(
            agent_id=self._agent_id,
            session_id=self._session_id,
            original_goal=self._original_goal,
            tool_name=tool_name,
            tool_input=arguments,
        )

        if verdict.verdict == Verdict.ALLOW:
            tool = self._tools.get(tool_name)
            if tool is None:
                return {"error": f"Unknown tool: {tool_name}"}
            return await tool.handler(**arguments)

        return {
            "error": "blocked",
            "verdict": verdict.verdict.value,
            "risk_score": verdict.risk_score,
            "reason": verdict.recommended_action,
        }
