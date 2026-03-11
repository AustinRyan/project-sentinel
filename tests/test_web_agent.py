"""Tests for the chat agent service."""
from __future__ import annotations

from janus.web.agent import ChatMessage, ChatResponse, ToolCallInfo


async def test_chat_response_structure() -> None:
    resp = ChatResponse(
        message="Hello",
        tool_calls=[],
        verdicts=[],
    )
    assert resp.message == "Hello"
    assert resp.tool_calls == []


async def test_chat_message_roles() -> None:
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


async def test_tool_call_info_fields() -> None:
    info = ToolCallInfo(
        tool_name="read_file",
        tool_input={"path": "/test.txt"},
        verdict="allow",
        risk_score=5.0,
        risk_delta=5.0,
        result={"content": "data"},
        reasons=["Permitted tool"],
    )
    assert info.tool_name == "read_file"
    assert info.verdict == "allow"
    assert info.result is not None
