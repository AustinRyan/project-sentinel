"""CrewAI integration adapter for Sentinel.

Usage::

    from sentinel.integrations.crewai import SentinelCrewTool

    tool = SentinelCrewTool(
        name="search", description="Search the web",
        fn=my_search_function,
        guardian=guardian, agent_id="agent-1", session_id="s-1",
    )
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from sentinel.core.decision import Verdict
from sentinel.core.guardian import Guardian


class SentinelCrewTool:
    """A CrewAI-compatible tool that intercepts calls with Guardian."""

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., Awaitable[str]],
        guardian: Guardian,
        agent_id: str,
        session_id: str,
        original_goal: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self._fn = fn
        self._guardian = guardian
        self._agent_id = agent_id
        self._session_id = session_id
        self._original_goal = original_goal

    async def run(self, tool_input: dict[str, Any]) -> str:
        """Execute with Guardian interception."""
        verdict = await self._guardian.wrap_tool_call(
            agent_id=self._agent_id,
            session_id=self._session_id,
            original_goal=self._original_goal,
            tool_name=self.name,
            tool_input=tool_input,
        )

        if verdict.verdict == Verdict.ALLOW:
            return await self._fn(**tool_input)

        return (
            f"BLOCKED by Sentinel (verdict={verdict.verdict.value}, "
            f"risk={verdict.risk_score:.1f}): {verdict.recommended_action}"
        )
