"""Tests for the LangChain integration adapter."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from janus.core.decision import SecurityVerdict, Verdict
from janus.integrations.langchain import JanusToolWrapper, janus_guard


async def test_janus_tool_wrapper_allows() -> None:
    mock_guardian = AsyncMock()
    mock_guardian.wrap_tool_call.return_value = SecurityVerdict(
        verdict=Verdict.ALLOW, risk_score=5.0, risk_delta=5.0,
    )

    mock_tool = MagicMock()
    mock_tool.name = "read_file"
    mock_tool.description = "Read a file"
    mock_tool.args_schema = None
    mock_tool.invoke = MagicMock(return_value="file contents")

    wrapper = JanusToolWrapper(
        tool=mock_tool,
        guardian=mock_guardian,
        agent_id="test-agent",
        session_id="test-session",
    )
    assert wrapper.name == "read_file"

    result = await wrapper.ainvoke({"path": "/test.txt"})
    assert result == "file contents"
    mock_guardian.wrap_tool_call.assert_called_once()


async def test_janus_tool_wrapper_blocks() -> None:
    mock_guardian = AsyncMock()
    mock_guardian.wrap_tool_call.return_value = SecurityVerdict(
        verdict=Verdict.BLOCK, risk_score=85.0, risk_delta=50.0,
        recommended_action="Blocked by security policy",
    )

    mock_tool = MagicMock()
    mock_tool.name = "execute_code"
    mock_tool.description = "Execute code"
    mock_tool.args_schema = None

    wrapper = JanusToolWrapper(
        tool=mock_tool,
        guardian=mock_guardian,
        agent_id="test-agent",
        session_id="test-session",
    )

    result = await wrapper.ainvoke({"code": "rm -rf /"})
    assert "BLOCKED" in result or "blocked" in result.lower()
    mock_tool.invoke.assert_not_called()


async def test_janus_guard_wraps_list() -> None:
    mock_guardian = AsyncMock()
    tools = [MagicMock(name=f"tool_{i}") for i in range(3)]
    for t in tools:
        t.description = "test"
        t.args_schema = None

    wrapped = janus_guard(
        tools=tools,
        guardian=mock_guardian,
        agent_id="test-agent",
        session_id="test-session",
    )
    assert len(wrapped) == 3
    assert all(isinstance(w, JanusToolWrapper) for w in wrapped)
