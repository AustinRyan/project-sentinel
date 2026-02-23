"""Tests for the OpenAI integration adapter."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from sentinel.core.decision import SecurityVerdict, Verdict
from sentinel.integrations.openai import SentinelFunctionProxy


async def test_openai_proxy_allows() -> None:
    mock_guardian = AsyncMock()
    mock_guardian.wrap_tool_call.return_value = SecurityVerdict(
        verdict=Verdict.ALLOW, risk_score=5.0, risk_delta=5.0,
    )

    async def read_file(path: str) -> str:
        return f"Contents of {path}"

    proxy = SentinelFunctionProxy(
        guardian=mock_guardian,
        agent_id="test-agent",
        session_id="test-session",
        functions={"read_file": read_file},
    )

    result = await proxy.execute("read_file", json.dumps({"path": "/test.txt"}))
    assert result.allowed
    assert "Contents of" in result.output


async def test_openai_proxy_blocks() -> None:
    mock_guardian = AsyncMock()
    mock_guardian.wrap_tool_call.return_value = SecurityVerdict(
        verdict=Verdict.BLOCK, risk_score=90.0, risk_delta=50.0,
        recommended_action="Blocked",
    )

    proxy = SentinelFunctionProxy(
        guardian=mock_guardian,
        agent_id="test-agent",
        session_id="test-session",
        functions={"execute_code": AsyncMock()},
    )

    result = await proxy.execute("execute_code", json.dumps({"code": "rm -rf /"}))
    assert not result.allowed
    assert "block" in result.output.lower()
