"""Janus — Autonomous Security Layer for AI Agents.

Usage::

    from janus import Guardian, JanusConfig, AgentIdentity, AgentRole

    config = JanusConfig()
    guardian = await Guardian.from_config(config, registry, session_store)

    verdict = await guardian.wrap_tool_call(
        agent_id="my-agent",
        session_id="session-123",
        original_goal="Summarize quarterly earnings",
        tool_name="read_file",
        tool_input={"path": "/reports/q4.pdf"},
    )

    if verdict.verdict == Verdict.ALLOW:
        result = execute_tool("read_file", {"path": "/reports/q4.pdf"})
    else:
        print(f"Blocked: {verdict.recommended_action}")
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("janus-security")
except PackageNotFoundError:
    __version__ = "0.1.0"

from janus.config import JanusConfig
from janus.core.decision import (
    CheckResult,
    PipelineContext,
    SecurityVerdict,
    ToolCallRequest,
    Verdict,
)
from janus.core.guardian import Guardian
from janus.identity.agent import AgentIdentity, AgentRole, ToolPermission
from janus.identity.registry import AgentRegistry
from janus.integrations import GuardResult, Janus, create_janus

__all__ = [
    "__version__",
    "AgentIdentity",
    "AgentRegistry",
    "AgentRole",
    "CheckResult",
    "create_janus",
    "GuardResult",
    "Guardian",
    "Janus",
    "PipelineContext",
    "SecurityVerdict",
    "JanusConfig",
    "ToolCallRequest",
    "ToolPermission",
    "Verdict",
]
