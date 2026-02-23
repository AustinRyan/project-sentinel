"""Chat agent service that bridges Claude Sonnet with Guardian interception."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic
import structlog

from sentinel.core.decision import Verdict
from sentinel.core.guardian import Guardian
from sentinel.web.events import EventBroadcaster, SecurityEvent
from sentinel.web.tools import MockToolExecutor

logger = structlog.get_logger()


@dataclass
class ChatMessage:
    """A single message in the conversation."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class ToolCallInfo:
    """Record of a tool call and its verdict."""

    tool_name: str
    tool_input: dict[str, Any]
    verdict: str
    risk_score: float
    risk_delta: float
    result: dict[str, Any] | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class ChatResponse:
    """Response from the chat agent."""

    message: str
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)


class ChatAgent:
    """Manages a conversation with Claude Sonnet, intercepting tool calls via Guardian."""

    def __init__(
        self,
        guardian: Guardian,
        broadcaster: EventBroadcaster,
        agent_id: str,
        session_id: str,
        original_goal: str = "",
        api_key: str | None = None,
    ) -> None:
        self._guardian = guardian
        self._broadcaster = broadcaster
        self._agent_id = agent_id
        self._session_id = session_id
        self._original_goal = original_goal
        self._tool_executor = MockToolExecutor()
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._history: list[dict[str, Any]] = []
        self._model = "claude-sonnet-4-6-20250220"

    async def chat(self, user_message: str) -> ChatResponse:
        """Process a user message through Claude + Guardian pipeline."""
        if not self._original_goal:
            self._original_goal = user_message

        self._history.append({"role": "user", "content": user_message})

        tool_calls: list[ToolCallInfo] = []
        messages = list(self._history)

        max_rounds = 10
        for _ in range(max_rounds):
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=(
                    "You are a helpful AI assistant with access to tools. "
                    "Use the tools when needed to complete the user's request. "
                    "Be concise in your responses."
                ),
                tools=self._tool_executor.get_tool_definitions(),  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
            )

            has_tool_use = any(
                block.type == "tool_use" for block in response.content
            )

            if not has_tool_use:
                text = "".join(
                    getattr(block, "text", "")
                    for block in response.content
                    if block.type == "text"
                )
                self._history.append(
                    {"role": "assistant", "content": text}
                )
                return ChatResponse(
                    message=text,
                    tool_calls=tool_calls,
                )

            assistant_content: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({
                        "type": "text",
                        "text": getattr(block, "text", ""),
                    })
                elif block.type == "tool_use":
                    block_id: str = getattr(block, "id", "")
                    block_name: str = getattr(block, "name", "")
                    raw_input = getattr(block, "input", {})
                    block_input: dict[str, Any] = (
                        raw_input if isinstance(raw_input, dict) else {}
                    )
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block_id,
                        "name": block_name,
                        "input": block_input,
                    })

                    verdict = await self._guardian.wrap_tool_call(
                        agent_id=self._agent_id,
                        session_id=self._session_id,
                        original_goal=self._original_goal,
                        tool_name=block_name,
                        tool_input=block_input,
                        conversation_history=self._history,
                    )

                    await self._broadcaster.publish(SecurityEvent(
                        event_type="verdict",
                        session_id=self._session_id,
                        data={
                            "verdict": verdict.verdict.value,
                            "risk_score": verdict.risk_score,
                            "risk_delta": verdict.risk_delta,
                            "tool_name": block_name,
                            "tool_input": block_input,
                            "reasons": verdict.reasons,
                            "drift_score": verdict.drift_score,
                            "itdr_signals": verdict.itdr_signals,
                            "recommended_action": verdict.recommended_action,
                            "trace_id": verdict.trace_id,
                        },
                    ))

                    if verdict.verdict == Verdict.ALLOW:
                        result = await self._tool_executor.execute(
                            block_name,
                            block_input,
                        )
                        tool_call_info = ToolCallInfo(
                            tool_name=block_name,
                            tool_input=block_input,
                            verdict="allow",
                            risk_score=verdict.risk_score,
                            risk_delta=verdict.risk_delta,
                            result=result,
                            reasons=verdict.reasons,
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block_id,
                            "content": json.dumps(result),
                        })
                    else:
                        denial_msg = (
                            f"Tool '{block_name}' was DENIED. "
                            f"Verdict: {verdict.verdict.value}. "
                            f"Reason: {verdict.recommended_action}"
                        )
                        tool_call_info = ToolCallInfo(
                            tool_name=block_name,
                            tool_input=block_input,
                            verdict=verdict.verdict.value,
                            risk_score=verdict.risk_score,
                            risk_delta=verdict.risk_delta,
                            reasons=verdict.reasons,
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block_id,
                            "content": denial_msg,
                            "is_error": True,
                        })

                    tool_calls.append(tool_call_info)

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        self._history.append({
            "role": "assistant",
            "content": "I reached the maximum number of tool call rounds.",
        })
        return ChatResponse(
            message="I reached the maximum number of tool call rounds.",
            tool_calls=tool_calls,
        )
