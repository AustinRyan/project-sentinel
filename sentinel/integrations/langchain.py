"""LangChain integration adapter for Sentinel.

Wraps any LangChain BaseTool with Guardian interception.

Usage::

    from langchain_core.tools import Tool
    from sentinel.integrations.langchain import sentinel_guard

    tools = [Tool(name="read", func=read_fn, description="Read files")]
    guarded = sentinel_guard(tools, guardian, "agent-1", "session-1")
    # Use `guarded` in your LangChain agent instead of `tools`
"""
from __future__ import annotations

from typing import Any

from sentinel.core.decision import Verdict
from sentinel.core.guardian import Guardian


class SentinelToolWrapper:
    """Wraps a LangChain tool with Guardian interception."""

    def __init__(
        self,
        tool: Any,
        guardian: Guardian,
        agent_id: str,
        session_id: str,
        original_goal: str = "",
    ) -> None:
        self._tool = tool
        self._guardian = guardian
        self._agent_id = agent_id
        self._session_id = session_id
        self._original_goal = original_goal

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def args_schema(self) -> Any:
        return self._tool.args_schema

    async def ainvoke(self, tool_input: dict[str, Any]) -> Any:
        """Intercept with Guardian, then execute if allowed."""
        verdict = await self._guardian.wrap_tool_call(
            agent_id=self._agent_id,
            session_id=self._session_id,
            original_goal=self._original_goal,
            tool_name=self.name,
            tool_input=tool_input,
        )

        if verdict.verdict == Verdict.ALLOW:
            return self._tool.invoke(tool_input)

        return (
            f"BLOCKED by Sentinel (verdict={verdict.verdict.value}, "
            f"risk={verdict.risk_score:.1f}): {verdict.recommended_action}"
        )

    def invoke(self, tool_input: dict[str, Any]) -> Any:
        """Synchronous invoke — raises if tool is blocked."""
        import asyncio
        return asyncio.run(self.ainvoke(tool_input))


def sentinel_guard(
    tools: list[Any],
    guardian: Guardian,
    agent_id: str,
    session_id: str,
    original_goal: str = "",
) -> list[SentinelToolWrapper]:
    """Wrap a list of LangChain tools with Sentinel Guardian interception."""
    return [
        SentinelToolWrapper(
            tool=t,
            guardian=guardian,
            agent_id=agent_id,
            session_id=session_id,
            original_goal=original_goal,
        )
        for t in tools
    ]
