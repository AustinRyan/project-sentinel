"""Tests for the CrewAI integration adapter."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from sentinel.core.decision import SecurityVerdict, Verdict
from sentinel.integrations.crewai import SentinelCrewTool


async def test_crewai_tool_allows() -> None:
    mock_guardian = AsyncMock()
    mock_guardian.wrap_tool_call.return_value = SecurityVerdict(
        verdict=Verdict.ALLOW, risk_score=5.0, risk_delta=5.0,
    )

    async def my_tool_fn(query: str) -> str:
        return f"Result for {query}"

    tool = SentinelCrewTool(
        name="search",
        description="Search the web",
        fn=my_tool_fn,
        guardian=mock_guardian,
        agent_id="test-agent",
        session_id="test-session",
    )

    result = await tool.run({"query": "test"})
    assert "Result for" in result


async def test_crewai_tool_blocks() -> None:
    mock_guardian = AsyncMock()
    mock_guardian.wrap_tool_call.return_value = SecurityVerdict(
        verdict=Verdict.BLOCK, risk_score=90.0, risk_delta=50.0,
        recommended_action="Blocked",
    )

    async def my_tool_fn(query: str) -> str:
        return "should not run"

    tool = SentinelCrewTool(
        name="execute_code",
        description="Execute code",
        fn=my_tool_fn,
        guardian=mock_guardian,
        agent_id="test-agent",
        session_id="test-session",
    )

    result = await tool.run({"code": "rm -rf /"})
    assert "BLOCKED" in result
