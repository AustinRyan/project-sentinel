"""OpenAI Assistants/Function Calling integration adapter for Sentinel.

Sits between OpenAI's function calling response and actual execution.

Usage::

    from sentinel.integrations.openai import SentinelFunctionProxy

    proxy = SentinelFunctionProxy(
        guardian=guardian, agent_id="a-1", session_id="s-1",
        functions={"read_file": read_file_fn},
    )
    result = await proxy.execute("read_file", '{"path": "/test.txt"}')
    if result.allowed:
        # feed result.output back to OpenAI
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from sentinel.core.decision import Verdict
from sentinel.core.guardian import Guardian


@dataclass
class FunctionResult:
    """Result of a function execution through Sentinel."""

    allowed: bool
    output: str
    verdict: str
    risk_score: float


class SentinelFunctionProxy:
    """Proxy for OpenAI function calling with Guardian interception."""

    def __init__(
        self,
        guardian: Guardian,
        agent_id: str,
        session_id: str,
        functions: dict[str, Callable[..., Awaitable[Any]]],
        original_goal: str = "",
    ) -> None:
        self._guardian = guardian
        self._agent_id = agent_id
        self._session_id = session_id
        self._functions = functions
        self._original_goal = original_goal

    async def execute(
        self, function_name: str, arguments: str
    ) -> FunctionResult:
        """Execute a function call with Guardian interception.

        Args:
            function_name: Name of the function to call.
            arguments: JSON string of function arguments (as OpenAI sends them).
        """
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            args = {"raw": arguments}

        verdict = await self._guardian.wrap_tool_call(
            agent_id=self._agent_id,
            session_id=self._session_id,
            original_goal=self._original_goal,
            tool_name=function_name,
            tool_input=args,
        )

        if verdict.verdict == Verdict.ALLOW:
            fn = self._functions.get(function_name)
            if fn is None:
                return FunctionResult(
                    allowed=False,
                    output=f"Unknown function: {function_name}",
                    verdict="error",
                    risk_score=verdict.risk_score,
                )
            result = await fn(**args)
            return FunctionResult(
                allowed=True,
                output=str(result),
                verdict="allow",
                risk_score=verdict.risk_score,
            )

        return FunctionResult(
            allowed=False,
            output=(
                f"BLOCKED by Sentinel (verdict={verdict.verdict.value}, "
                f"risk={verdict.risk_score:.1f}): {verdict.recommended_action}"
            ),
            verdict=verdict.verdict.value,
            risk_score=verdict.risk_score,
        )
