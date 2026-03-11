"""LangChain integration adapter for Janus.

Wraps any LangChain BaseTool with Guardian interception.

Usage::

    from langchain_core.tools import Tool
    from janus.integrations.langchain import janus_guard

    tools = [Tool(name="read", func=read_fn, description="Read files")]
    guarded = janus_guard(tools, guardian, "agent-1", "session-1")
    # Use `guarded` in your LangChain agent instead of `tools`
"""
from __future__ import annotations

from typing import Any

import structlog

from janus.core.approval import ApprovalManager, needs_human_review
from janus.core.decision import Verdict
from janus.core.guardian import Guardian

logger = structlog.get_logger()


class JanusToolWrapper:
    """Wraps a LangChain tool with Guardian interception."""

    def __init__(
        self,
        tool: Any,
        guardian: Guardian,
        agent_id: str,
        session_id: str,
        original_goal: str = "",
        approval_manager: ApprovalManager | None = None,
    ) -> None:
        self._tool = tool
        self._guardian = guardian
        self._agent_id = agent_id
        self._session_id = session_id
        self._original_goal = original_goal
        self._approval_manager = approval_manager

    @property
    def name(self) -> str:
        return str(self._tool.name)

    @property
    def description(self) -> str:
        return str(self._tool.description)

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

        # Check if this needs human review
        approval_id: str | None = None
        if self._approval_manager is not None:
            cr_dicts = [
                {
                    "check_name": cr.check_name,
                    "passed": cr.passed,
                    "risk_contribution": cr.risk_contribution,
                    "reason": cr.reason,
                    "metadata": cr.metadata,
                    "force_verdict": cr.force_verdict.value if cr.force_verdict else None,
                }
                for cr in verdict.check_results
            ]
            if needs_human_review(verdict.verdict.value, cr_dicts):
                try:
                    request = await self._approval_manager.create(
                        session_id=self._session_id,
                        agent_id=self._agent_id,
                        tool_name=self.name,
                        tool_input=tool_input,
                        original_goal=self._original_goal,
                        verdict=verdict.verdict.value,
                        risk_score=verdict.risk_score,
                        risk_delta=verdict.risk_delta,
                        reasons=verdict.reasons,
                        check_results=cr_dicts,
                        trace_id=verdict.trace_id,
                    )
                    approval_id = request.id
                except Exception:
                    logger.exception("langchain_approval_create_error")

        pending_note = ""
        if approval_id:
            pending_note = f" Approval ID: {approval_id}. A human reviewer has been notified."

        return (
            f"BLOCKED by Janus (verdict={verdict.verdict.value}, "
            f"risk={verdict.risk_score:.1f}): {verdict.recommended_action}.{pending_note}"
        )

    def invoke(self, tool_input: dict[str, Any]) -> Any:
        """Synchronous invoke — raises if tool is blocked."""
        import asyncio
        return asyncio.run(self.ainvoke(tool_input))


def janus_guard(
    tools: list[Any],
    guardian: Guardian,
    agent_id: str,
    session_id: str,
    original_goal: str = "",
    approval_manager: ApprovalManager | None = None,
) -> list[JanusToolWrapper]:
    """Wrap a list of LangChain tools with Janus Guardian interception."""
    return [
        JanusToolWrapper(
            tool=t,
            guardian=guardian,
            agent_id=agent_id,
            session_id=session_id,
            original_goal=original_goal,
            approval_manager=approval_manager,
        )
        for t in tools
    ]


async def create_langchain_guard(
    tools: list[Any],
    *,
    agent_id: str = "langchain-agent",
    agent_name: str = "LangChain Agent",
    agent_role: str = "code",
    permissions: list[str] | None = None,
    session_id: str | None = None,
    original_goal: str = "",
    config: Any | None = None,
    db_path: str | None = None,
    api_key: str | None = None,
) -> list[JanusToolWrapper]:
    """One-call factory: wrap LangChain tools with Janus protection.

    Usage::

        from janus.integrations.langchain import create_langchain_guard

        guarded = await create_langchain_guard(
            [search_tool, db_tool],
            agent_id="my-agent",
            original_goal="Research task",
        )
        agent = create_react_agent(llm, guarded)
    """
    from janus.integrations import create_janus

    janus = await create_janus(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_role=agent_role,
        permissions=permissions,
        session_id=session_id,
        original_goal=original_goal,
        config=config,
        db_path=db_path,
        api_key=api_key,
    )

    return janus_guard(
        tools,
        guardian=janus.guardian,
        agent_id=agent_id,
        session_id=janus.session_id,
        original_goal=original_goal,
        approval_manager=janus.approval_manager,
    )
