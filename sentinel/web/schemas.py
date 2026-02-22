"""Pydantic schemas for the REST API."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ToolCallOut(BaseModel):
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    verdict: str
    risk_score: float
    risk_delta: float
    result: dict[str, Any] | None = None
    reasons: list[str] = Field(default_factory=list)


class ChatResponseOut(BaseModel):
    message: str
    tool_calls: list[ToolCallOut] = Field(default_factory=list)
    session_id: str


class SessionCreateRequest(BaseModel):
    agent_id: str = "demo-agent"
    original_goal: str = ""


class SessionOut(BaseModel):
    session_id: str
    agent_id: str
    original_goal: str
    risk_score: float


class AgentOut(BaseModel):
    agent_id: str
    name: str
    role: str
    permissions: list[str]
    is_locked: bool


class HealthOut(BaseModel):
    status: str
    total_requests: int = 0
    error_rate: float = 0.0
    circuit_breaker: str = "closed"


class TraceOut(BaseModel):
    trace_id: str
    session_id: str
    agent_id: str
    tool_name: str
    verdict: str
    risk_score: float
    risk_delta: float
    explanation: str
    timestamp: str
    reasons: list[str] = Field(default_factory=list)
