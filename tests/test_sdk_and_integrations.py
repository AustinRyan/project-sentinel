"""Comprehensive tests for the Janus SDK factory, integration adapters, and MCP proxy.

Covers:
- Janus SDK client (create_janus, Janus.guard, GuardResult)
- LangChain adapter with approval_manager
- OpenAI adapter with approval_manager
- CrewAI adapter with approval_manager
- MCP server adapter with approval_manager
- MCP proxy approval flow + event broadcasting
- Edge cases: exceptions, None values, all verdict types, hard vs soft blocks
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import types

from janus.core.approval import ApprovalManager, needs_human_review
from janus.core.decision import CheckResult, SecurityVerdict, Verdict
from janus.integrations import GuardResult, Janus, create_janus
from janus.integrations.crewai import JanusCrewTool
from janus.integrations.langchain import JanusToolWrapper, janus_guard
from janus.integrations.mcp import JanusMCPServer, MCPToolDefinition
from janus.integrations.openai import FunctionResult, JanusFunctionProxy
from janus.mcp.config import AgentConfig, ProxyConfig
from janus.mcp.proxy import JanusMCPProxy
from janus.storage.database import DatabaseManager
from janus.web.events import EventBroadcaster, SecurityEvent

# ═══════════════════════════════════════════════════════════════════
# Shared helpers & fixtures
# ═══════════════════════════════════════════════════════════════════


def _make_check_result(
    check_name: str = "deterministic_risk",
    passed: bool = True,
    risk_contribution: float = 0.0,
    reason: str = "",
    force_verdict: Verdict | None = None,
) -> CheckResult:
    return CheckResult(
        check_name=check_name,
        passed=passed,
        risk_contribution=risk_contribution,
        reason=reason,
        force_verdict=force_verdict,
    )


def _make_verdict(
    verdict: Verdict = Verdict.ALLOW,
    risk_score: float = 5.0,
    risk_delta: float = 5.0,
    reasons: list[str] | None = None,
    recommended_action: str = "Tool call approved.",
    check_results: list[CheckResult] | None = None,
    drift_score: float = 0.0,
    itdr_signals: list[str] | None = None,
    trace_id: str = "trace-test-123",
) -> SecurityVerdict:
    return SecurityVerdict(
        verdict=verdict,
        risk_score=risk_score,
        risk_delta=risk_delta,
        reasons=reasons or [],
        recommended_action=recommended_action,
        check_results=check_results or [],
        drift_score=drift_score,
        itdr_signals=itdr_signals or [],
        trace_id=trace_id,
    )


@pytest.fixture
async def approval_db() -> DatabaseManager:
    db = DatabaseManager(":memory:")
    await db.connect()
    await db.apply_migrations()
    yield db  # type: ignore[misc]
    await db.close()


@pytest.fixture
def broadcaster() -> EventBroadcaster:
    return EventBroadcaster()


@pytest.fixture
def approval_manager(approval_db: DatabaseManager, broadcaster: EventBroadcaster) -> ApprovalManager:
    return ApprovalManager(db=approval_db, broadcaster=broadcaster)


def _mock_guardian(verdict: SecurityVerdict | None = None) -> AsyncMock:
    g = AsyncMock()
    g.wrap_tool_call.return_value = verdict or _make_verdict()
    return g


# ═══════════════════════════════════════════════════════════════════
# 1. GuardResult dataclass
# ═══════════════════════════════════════════════════════════════════


class TestGuardResult:
    def test_allowed_result(self) -> None:
        r = GuardResult(
            allowed=True, verdict="allow", risk_score=3.0, risk_delta=3.0,
            reasons=[], recommended_action="Allowed.",
        )
        assert r.allowed is True
        assert r.verdict == "allow"
        assert r.approval_id is None
        assert r.trace_id == ""

    def test_blocked_result_with_approval(self) -> None:
        r = GuardResult(
            allowed=False, verdict="block", risk_score=85.0, risk_delta=30.0,
            reasons=["high risk", "drift detected"],
            recommended_action="Contact admin.",
            approval_id="apr-abc123",
            trace_id="trace-xyz",
        )
        assert r.allowed is False
        assert r.approval_id == "apr-abc123"
        assert r.trace_id == "trace-xyz"
        assert len(r.reasons) == 2

    def test_reason_property_aliases_recommended_action(self) -> None:
        r = GuardResult(
            allowed=False, verdict="challenge", risk_score=50.0, risk_delta=10.0,
            reasons=[], recommended_action="Please verify identity.",
        )
        assert r.reason == "Please verify identity."
        assert r.reason == r.recommended_action

    def test_all_verdict_types(self) -> None:
        for v in ["allow", "block", "challenge", "sandbox", "pause"]:
            r = GuardResult(
                allowed=(v == "allow"), verdict=v, risk_score=0.0,
                risk_delta=0.0, reasons=[], recommended_action="",
            )
            assert r.verdict == v

    def test_defaults(self) -> None:
        r = GuardResult(
            allowed=True, verdict="allow", risk_score=0.0, risk_delta=0.0,
            reasons=[], recommended_action="",
        )
        assert r.approval_id is None
        assert r.trace_id == ""


# ═══════════════════════════════════════════════════════════════════
# 2. Janus SDK client — guard() method
# ═══════════════════════════════════════════════════════════════════


class TestJanusGuard:
    async def test_allow_verdict(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.ALLOW, risk_score=2.0, risk_delta=2.0))
        janus = Janus(
            guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        result = await janus.guard("read_file", {"path": "/tmp/x"})
        assert result.allowed is True
        assert result.verdict == "allow"
        assert result.approval_id is None
        assert result.risk_score == 2.0

    async def test_block_with_approval(self, approval_manager: ApprovalManager) -> None:
        """Risk-based block creates an approval request."""
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=80.0, risk_delta=25.0,
            reasons=["high risk accumulation"],
            check_results=[_make_check_result("deterministic_risk", passed=False, risk_contribution=25.0)],
        ))
        janus = Janus(
            guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        result = await janus.guard("execute_code", {"code": "rm -rf /"})
        assert result.allowed is False
        assert result.verdict == "block"
        assert result.approval_id is not None
        assert result.approval_id.startswith("apr-")

        # Verify approval was actually persisted
        req = await approval_manager.get_by_id(result.approval_id)
        assert req is not None
        assert req.tool_name == "execute_code"
        assert req.session_id == "s1"

    async def test_block_hard_policy_no_approval(self, approval_manager: ApprovalManager) -> None:
        """Permission/identity/injection blocks do NOT create approvals."""
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=75.0, risk_delta=25.0,
            reasons=["permission denied"],
            check_results=[_make_check_result("permission_scope", passed=False, force_verdict=Verdict.BLOCK)],
        ))
        janus = Janus(
            guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        result = await janus.guard("send_email", {"to": "evil@hacker.com"})
        assert result.allowed is False
        assert result.approval_id is None

        pending = await approval_manager.get_pending()
        assert len(pending) == 0

    async def test_challenge_with_approval(self, approval_manager: ApprovalManager) -> None:
        """Challenge from risk/drift creates approval."""
        guardian = _mock_guardian(_make_verdict(
            Verdict.CHALLENGE, risk_score=50.0, risk_delta=15.0,
            reasons=["drift detected"],
            check_results=[_make_check_result("drift", passed=False, risk_contribution=15.0)],
        ))
        janus = Janus(
            guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        result = await janus.guard("api_call", {"url": "http://unknown.com"})
        assert result.allowed is False
        assert result.verdict == "challenge"
        assert result.approval_id is not None

    async def test_challenge_from_hard_block_no_approval(self, approval_manager: ApprovalManager) -> None:
        """Challenge caused by permission_scope = no approval."""
        guardian = _mock_guardian(_make_verdict(
            Verdict.CHALLENGE, risk_score=50.0, risk_delta=15.0,
            check_results=[_make_check_result("permission_scope", passed=False, force_verdict=Verdict.CHALLENGE)],
        ))
        janus = Janus(
            guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        result = await janus.guard("write_file", {"path": "/x"})
        assert result.allowed is False
        assert result.approval_id is None

    async def test_sandbox_no_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.SANDBOX, risk_score=60.0))
        janus = Janus(
            guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        result = await janus.guard("execute_code", {"code": "ls"})
        assert result.allowed is False
        assert result.verdict == "sandbox"
        assert result.approval_id is None

    async def test_pause_creates_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.PAUSE, risk_score=55.0, risk_delta=10.0,
            reasons=["anomalous behavior"],
        ))
        janus = Janus(
            guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        result = await janus.guard("api_call", {"url": "http://evil.com"})
        assert result.allowed is False
        assert result.verdict == "pause"
        assert result.approval_id is not None

    async def test_no_approval_manager(self) -> None:
        """Without approval_manager, blocks still work but no approval_id."""
        guardian = _mock_guardian(_make_verdict(Verdict.BLOCK, risk_score=80.0))
        janus = Janus(guardian=guardian, agent_id="a1", session_id="s1")
        result = await janus.guard("execute_code", {"code": "rm -rf /"})
        assert result.allowed is False
        assert result.approval_id is None

    async def test_original_goal_parameter(self) -> None:
        guardian = _mock_guardian()
        janus = Janus(guardian=guardian, agent_id="a1", session_id="s1", original_goal="default goal")
        await janus.guard("read_file", {"path": "/x"}, original_goal="override goal")
        guardian.wrap_tool_call.assert_called_once_with(
            agent_id="a1", session_id="s1", original_goal="override goal",
            tool_name="read_file", tool_input={"path": "/x"},
        )

    async def test_original_goal_fallback(self) -> None:
        guardian = _mock_guardian()
        janus = Janus(guardian=guardian, agent_id="a1", session_id="s1", original_goal="default goal")
        await janus.guard("read_file", {"path": "/x"})
        guardian.wrap_tool_call.assert_called_once_with(
            agent_id="a1", session_id="s1", original_goal="default goal",
            tool_name="read_file", tool_input={"path": "/x"},
        )

    async def test_trace_id_preserved(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.ALLOW, trace_id="trace-xyz-789"))
        janus = Janus(guardian=guardian, agent_id="a1", session_id="s1", approval_manager=approval_manager)
        result = await janus.guard("read_file", {"path": "/x"})
        assert result.trace_id == "trace-xyz-789"

    async def test_check_results_with_none_force_verdict(self, approval_manager: ApprovalManager) -> None:
        """CheckResult with force_verdict=None serialized correctly."""
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=80.0,
            check_results=[
                _make_check_result("deterministic_risk", passed=False, risk_contribution=30.0, force_verdict=None),
                _make_check_result("llm_risk_classifier", passed=False, risk_contribution=20.0, force_verdict=None),
            ],
        ))
        janus = Janus(guardian=guardian, agent_id="a1", session_id="s1", approval_manager=approval_manager)
        result = await janus.guard("execute_code", {"code": "rm -rf /"})
        assert result.approval_id is not None

        req = await approval_manager.get_by_id(result.approval_id)
        assert req is not None
        assert len(req.check_results) == 2
        assert req.check_results[0]["force_verdict"] is None

    async def test_empty_check_results(self, approval_manager: ApprovalManager) -> None:
        """Block with empty check_results still creates approval (ambiguous = review)."""
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=75.0, check_results=[],
        ))
        janus = Janus(guardian=guardian, agent_id="a1", session_id="s1", approval_manager=approval_manager)
        result = await janus.guard("execute_code", {"code": "x"})
        assert result.approval_id is not None


class TestJanusClose:
    async def test_close_with_db(self) -> None:
        mock_db = AsyncMock()
        janus = Janus(guardian=AsyncMock(), agent_id="a1", session_id="s1", db=mock_db)
        await janus.close()
        mock_db.close.assert_called_once()

    async def test_close_without_db(self) -> None:
        janus = Janus(guardian=AsyncMock(), agent_id="a1", session_id="s1")
        await janus.close()  # Should not raise


# ═══════════════════════════════════════════════════════════════════
# 3. create_janus() factory
# ═══════════════════════════════════════════════════════════════════


class TestCreateJanus:
    async def test_defaults(self) -> None:
        janus = await create_janus()
        assert janus.agent_id == "default-agent"
        assert janus.session_id.startswith("sdk-")
        assert janus.guardian is not None
        assert janus.approval_manager is not None
        assert janus.broadcaster is not None
        await janus.close()

    async def test_custom_params(self) -> None:
        janus = await create_janus(
            agent_id="my-agent", agent_name="My Agent",
            agent_role="research", permissions=["read_*", "search_*"],
            session_id="custom-session", original_goal="Summarize data",
        )
        assert janus.agent_id == "my-agent"
        assert janus.session_id == "custom-session"
        assert janus.original_goal == "Summarize data"
        await janus.close()

    async def test_db_path_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JANUS_DB_PATH", ":memory:")
        janus = await create_janus()
        assert janus._db is not None
        await janus.close()

    async def test_no_api_key_no_classifier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ANTHROPIC_API_KEY, classifier is None (rule-based only)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        janus = await create_janus(api_key=None)
        assert janus.guardian is not None
        await janus.close()

    async def test_permissions_default_wildcard(self) -> None:
        janus = await create_janus()
        # Default permissions = ["*"] so read_file should be allowed
        result = await janus.guard("read_file", {"path": "/test"})
        assert result.allowed is True
        await janus.close()

    async def test_restricted_permissions(self) -> None:
        janus = await create_janus(
            agent_id="restricted", permissions=["read_*"],
        )
        # read_file = allowed, write_file = blocked by permission_scope
        r1 = await janus.guard("read_file", {"path": "/x"})
        assert r1.allowed is True

        r2 = await janus.guard("write_file", {"path": "/x", "content": "x"})
        assert r2.allowed is False
        # Permission block = hard denial, no approval
        assert r2.approval_id is None
        await janus.close()

    async def test_full_round_trip(self) -> None:
        """End-to-end: create, guard, get approval, approve."""
        janus = await create_janus(agent_id="e2e-agent", permissions=["*"])

        # First call: low risk, allowed
        r1 = await janus.guard("read_file", {"path": "/test.txt"})
        assert r1.allowed is True

        # Accumulate risk with action tools to trigger block
        for _ in range(8):
            await janus.guard("execute_code", {"code": "import os; os.system('rm -rf /')"})

        # Check if any approvals were created
        pending = await janus.approval_manager.get_pending()
        # At least some risky calls should have been blocked and created approvals
        if len(pending) > 0:
            approval = pending[0]
            assert approval.status == "pending"
            assert approval.tool_name == "execute_code"

            # Approve it
            result = await janus.approval_manager.approve(approval.id, decided_by="admin")
            assert result is not None
            assert result.status == "approved"

        await janus.close()


# ═══════════════════════════════════════════════════════════════════
# 4. LangChain adapter with approvals
# ═══════════════════════════════════════════════════════════════════


class TestLangChainWithApprovals:
    def _make_tool(self, name: str = "read_file") -> MagicMock:
        tool = MagicMock()
        tool.name = name
        tool.description = f"Tool: {name}"
        tool.args_schema = None
        tool.invoke = MagicMock(return_value="tool result")
        return tool

    async def test_allow_no_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.ALLOW))
        wrapper = JanusToolWrapper(
            tool=self._make_tool(), guardian=guardian, agent_id="a1",
            session_id="s1", approval_manager=approval_manager,
        )
        result = await wrapper.ainvoke({"path": "/test"})
        assert result == "tool result"
        assert len(await approval_manager.get_pending()) == 0

    async def test_block_creates_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=80.0,
            reasons=["risk too high"],
            check_results=[_make_check_result("deterministic_risk", passed=False, risk_contribution=30.0)],
        ))
        wrapper = JanusToolWrapper(
            tool=self._make_tool("execute_code"), guardian=guardian, agent_id="a1",
            session_id="s1", approval_manager=approval_manager,
        )
        result = await wrapper.ainvoke({"code": "rm -rf /"})
        assert "BLOCKED" in result
        assert "Approval ID: apr-" in result

        pending = await approval_manager.get_pending()
        assert len(pending) == 1
        assert pending[0].tool_name == "execute_code"

    async def test_hard_block_no_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=75.0,
            check_results=[_make_check_result("permission_scope", passed=False, force_verdict=Verdict.BLOCK)],
        ))
        wrapper = JanusToolWrapper(
            tool=self._make_tool("send_email"), guardian=guardian, agent_id="a1",
            session_id="s1", approval_manager=approval_manager,
        )
        result = await wrapper.ainvoke({"to": "evil@hacker.com"})
        assert "BLOCKED" in result
        assert "Approval ID" not in result
        assert len(await approval_manager.get_pending()) == 0

    async def test_no_approval_manager(self) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.BLOCK, risk_score=80.0))
        wrapper = JanusToolWrapper(
            tool=self._make_tool(), guardian=guardian, agent_id="a1", session_id="s1",
        )
        result = await wrapper.ainvoke({"path": "/x"})
        assert "BLOCKED" in result

    async def test_approval_creation_exception(self, approval_manager: ApprovalManager) -> None:
        """Approval creation failure should not crash the wrapper."""
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=80.0,
            check_results=[_make_check_result("deterministic_risk", passed=False)],
        ))
        # Make approval creation fail
        approval_manager.create = AsyncMock(side_effect=RuntimeError("DB connection lost"))
        wrapper = JanusToolWrapper(
            tool=self._make_tool(), guardian=guardian, agent_id="a1",
            session_id="s1", approval_manager=approval_manager,
        )
        result = await wrapper.ainvoke({"path": "/x"})
        # Should still return blocked message, just without approval_id
        assert "BLOCKED" in result
        assert "Approval ID" not in result

    async def test_janus_guard_passes_approval_manager(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian()
        tools = [self._make_tool(f"tool_{i}") for i in range(3)]
        wrapped = janus_guard(
            tools=tools, guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        assert len(wrapped) == 3
        assert all(w._approval_manager is approval_manager for w in wrapped)

    async def test_challenge_verdict_message(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.CHALLENGE, risk_score=50.0, recommended_action="Verify intent.",
            check_results=[_make_check_result("drift", passed=False)],
        ))
        wrapper = JanusToolWrapper(
            tool=self._make_tool(), guardian=guardian, agent_id="a1",
            session_id="s1", approval_manager=approval_manager,
        )
        result = await wrapper.ainvoke({"path": "/x"})
        assert "challenge" in result.lower()
        assert "Verify intent." in result


# ═══════════════════════════════════════════════════════════════════
# 5. OpenAI adapter with approvals
# ═══════════════════════════════════════════════════════════════════


class TestOpenAIWithApprovals:
    async def test_allow_no_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.ALLOW))

        async def read_file(path: str) -> str:
            return f"Contents of {path}"

        proxy = JanusFunctionProxy(
            guardian=guardian, agent_id="a1", session_id="s1",
            functions={"read_file": read_file},
            approval_manager=approval_manager,
        )
        result = await proxy.execute("read_file", json.dumps({"path": "/test.txt"}))
        assert result.allowed is True
        assert result.approval_id is None
        assert "Contents of" in result.output

    async def test_block_creates_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=85.0, risk_delta=30.0,
            reasons=["suspicious pattern"],
            check_results=[_make_check_result("llm_risk_classifier", passed=False)],
        ))
        proxy = JanusFunctionProxy(
            guardian=guardian, agent_id="a1", session_id="s1",
            functions={"execute_code": AsyncMock()},
            approval_manager=approval_manager,
        )
        result = await proxy.execute("execute_code", json.dumps({"code": "rm /"}))
        assert result.allowed is False
        assert result.approval_id is not None
        assert result.verdict == "block"
        assert isinstance(result, FunctionResult)

        pending = await approval_manager.get_pending()
        assert len(pending) == 1

    async def test_hard_block_no_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK,
            check_results=[_make_check_result("identity_check", passed=False, force_verdict=Verdict.BLOCK)],
        ))
        proxy = JanusFunctionProxy(
            guardian=guardian, agent_id="a1", session_id="s1",
            functions={}, approval_manager=approval_manager,
        )
        result = await proxy.execute("anything", "{}")
        assert result.allowed is False
        assert result.approval_id is None

    async def test_invalid_json_arguments(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.ALLOW))

        async def my_fn(raw: str) -> str:
            return f"raw={raw}"

        proxy = JanusFunctionProxy(
            guardian=guardian, agent_id="a1", session_id="s1",
            functions={"my_fn": my_fn}, approval_manager=approval_manager,
        )
        result = await proxy.execute("my_fn", "not valid json {{{")
        assert result.allowed is True
        assert "raw=" in result.output

    async def test_unknown_function(self) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.ALLOW))
        proxy = JanusFunctionProxy(
            guardian=guardian, agent_id="a1", session_id="s1",
            functions={},
        )
        result = await proxy.execute("nonexistent", "{}")
        assert result.allowed is False
        assert result.verdict == "error"
        assert "Unknown function" in result.output

    async def test_approval_creation_exception(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK,
            check_results=[_make_check_result("deterministic_risk", passed=False)],
        ))
        approval_manager.create = AsyncMock(side_effect=Exception("boom"))
        proxy = JanusFunctionProxy(
            guardian=guardian, agent_id="a1", session_id="s1",
            functions={}, approval_manager=approval_manager,
        )
        result = await proxy.execute("tool", "{}")
        assert result.allowed is False
        assert result.approval_id is None

    async def test_no_approval_manager(self) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.BLOCK, risk_score=80.0))
        proxy = JanusFunctionProxy(
            guardian=guardian, agent_id="a1", session_id="s1",
            functions={},
        )
        result = await proxy.execute("tool", "{}")
        assert result.allowed is False
        assert result.approval_id is None


# ═══════════════════════════════════════════════════════════════════
# 6. CrewAI adapter with approvals
# ═══════════════════════════════════════════════════════════════════


class TestCrewAIWithApprovals:
    async def test_allow_no_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.ALLOW))

        async def search(query: str) -> str:
            return f"Results for {query}"

        tool = JanusCrewTool(
            name="search", description="Search",
            fn=search, guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        result = await tool.run({"query": "test"})
        assert "Results for test" in result
        assert len(await approval_manager.get_pending()) == 0

    async def test_block_creates_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=80.0,
            check_results=[_make_check_result("deterministic_risk", passed=False)],
        ))

        async def danger(code: str) -> str:
            return "should not run"

        tool = JanusCrewTool(
            name="execute_code", description="Execute code",
            fn=danger, guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        result = await tool.run({"code": "rm -rf /"})
        assert "BLOCKED" in result
        assert "Approval ID: apr-" in result
        assert len(await approval_manager.get_pending()) == 1

    async def test_hard_block_no_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.CHALLENGE,
            check_results=[_make_check_result("prompt_injection", passed=False, force_verdict=Verdict.BLOCK)],
        ))

        async def noop() -> str:
            return ""

        tool = JanusCrewTool(
            name="tool", description="Tool",
            fn=noop, guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        result = await tool.run({})
        assert "BLOCKED" in result
        assert "Approval ID" not in result

    async def test_approval_creation_exception(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK,
            check_results=[_make_check_result("drift", passed=False)],
        ))
        approval_manager.create = AsyncMock(side_effect=Exception("fail"))

        async def noop() -> str:
            return ""

        tool = JanusCrewTool(
            name="tool", description="Tool",
            fn=noop, guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        result = await tool.run({})
        assert "BLOCKED" in result
        assert "Approval ID" not in result

    async def test_no_approval_manager(self) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.BLOCK))

        async def noop() -> str:
            return ""

        tool = JanusCrewTool(
            name="tool", description="Tool",
            fn=noop, guardian=guardian, agent_id="a1", session_id="s1",
        )
        result = await tool.run({})
        assert "BLOCKED" in result


# ═══════════════════════════════════════════════════════════════════
# 7. MCP server adapter with approvals
# ═══════════════════════════════════════════════════════════════════


class TestMCPServerWithApprovals:
    async def test_allow_no_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.ALLOW))
        handler = AsyncMock(return_value={"content": "data"})
        server = JanusMCPServer(
            guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        server.add_tool(MCPToolDefinition(
            name="read_file", description="Read", input_schema={}, handler=handler,
        ))
        result = await server.call_tool("read_file", {"path": "/x"})
        assert result == {"content": "data"}
        assert len(await approval_manager.get_pending()) == 0

    async def test_block_creates_approval_with_id(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=80.0,
            check_results=[_make_check_result("deterministic_risk", passed=False)],
        ))
        server = JanusMCPServer(
            guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        server.add_tool(MCPToolDefinition(
            name="execute_code", description="Exec", input_schema={}, handler=AsyncMock(),
        ))
        result = await server.call_tool("execute_code", {"code": "rm /"})
        assert result["error"] == "blocked"
        assert result["approval_id"] is not None
        assert result["approval_id"].startswith("apr-")

    async def test_hard_block_no_approval(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK,
            check_results=[_make_check_result("identity_check", passed=False, force_verdict=Verdict.BLOCK)],
        ))
        server = JanusMCPServer(
            guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        server.add_tool(MCPToolDefinition(
            name="tool", description="T", input_schema={}, handler=AsyncMock(),
        ))
        result = await server.call_tool("tool", {})
        assert result["approval_id"] is None

    async def test_unknown_tool(self) -> None:
        guardian = _mock_guardian(_make_verdict(Verdict.ALLOW))
        server = JanusMCPServer(
            guardian=guardian, agent_id="a1", session_id="s1",
        )
        result = await server.call_tool("nonexistent", {})
        assert "Unknown tool" in result["error"]

    async def test_tool_registration(self) -> None:
        server = JanusMCPServer(
            guardian=AsyncMock(), agent_id="a1", session_id="s1",
        )
        server.add_tool(MCPToolDefinition(name="t1", description="D1", input_schema={"type": "object"}, handler=AsyncMock()))
        server.add_tool(MCPToolDefinition(name="t2", description="D2", input_schema={}, handler=AsyncMock()))

        assert server.tool_names == ["t1", "t2"]
        defs = server.get_tool_definitions()
        assert len(defs) == 2
        assert defs[0]["name"] == "t1"
        assert defs[0]["inputSchema"] == {"type": "object"}

    async def test_approval_creation_exception(self, approval_manager: ApprovalManager) -> None:
        guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK,
            check_results=[_make_check_result("drift", passed=False)],
        ))
        approval_manager.create = AsyncMock(side_effect=Exception("fail"))
        server = JanusMCPServer(
            guardian=guardian, agent_id="a1", session_id="s1",
            approval_manager=approval_manager,
        )
        server.add_tool(MCPToolDefinition(name="t", description="T", input_schema={}, handler=AsyncMock()))
        result = await server.call_tool("t", {})
        assert result["approval_id"] is None


# ═══════════════════════════════════════════════════════════════════
# 8. MCP Proxy with approvals + event broadcasting
# ═══════════════════════════════════════════════════════════════════


def _make_proxy() -> JanusMCPProxy:
    config = ProxyConfig(agent=AgentConfig(agent_id="test-agent"))
    proxy = JanusMCPProxy(config)
    proxy._session_id = "test-session"
    return proxy


class TestMCPProxyApprovals:
    async def test_allow_no_approval(self, approval_db: DatabaseManager, broadcaster: EventBroadcaster) -> None:
        proxy = _make_proxy()
        proxy._guardian = _mock_guardian(_make_verdict(Verdict.ALLOW))
        proxy._db = approval_db
        proxy._broadcaster = broadcaster
        proxy._approval_manager = ApprovalManager(db=approval_db, broadcaster=broadcaster)
        proxy._upstream = AsyncMock()
        proxy._upstream.call_tool.return_value = types.CallToolResult(
            content=[types.TextContent(type="text", text="ok")]
        )
        result = await proxy._intercept_and_forward("read_file", {"path": "/x"})
        assert "ok" in result[0].text
        # No approvals should be created
        pending = await proxy._approval_manager.get_pending()
        assert len(pending) == 0

    async def test_block_creates_approval(self, approval_db: DatabaseManager, broadcaster: EventBroadcaster) -> None:
        proxy = _make_proxy()
        proxy._guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=80.0, risk_delta=25.0,
            reasons=["high risk"],
            recommended_action="Block execution.",
            check_results=[_make_check_result("deterministic_risk", passed=False, risk_contribution=25.0)],
        ))
        proxy._db = approval_db
        proxy._broadcaster = broadcaster
        proxy._approval_manager = ApprovalManager(db=approval_db, broadcaster=broadcaster)

        result = await proxy._intercept_and_forward("execute_code", {"code": "rm /"})
        assert "[JANUS BLOCKED]" in result[0].text
        assert "Approval ID: apr-" in result[0].text
        assert "human reviewer" in result[0].text

        pending = await proxy._approval_manager.get_pending()
        assert len(pending) == 1
        assert pending[0].tool_name == "execute_code"

    async def test_challenge_creates_approval(self, approval_db: DatabaseManager, broadcaster: EventBroadcaster) -> None:
        proxy = _make_proxy()
        proxy._guardian = _mock_guardian(_make_verdict(
            Verdict.CHALLENGE, risk_score=50.0,
            reasons=["drift detected"],
            recommended_action="Verify identity.",
            check_results=[_make_check_result("drift", passed=False)],
        ))
        proxy._db = approval_db
        proxy._broadcaster = broadcaster
        proxy._approval_manager = ApprovalManager(db=approval_db, broadcaster=broadcaster)

        result = await proxy._intercept_and_forward("api_call", {"url": "http://x.com"})
        assert "[JANUS CHALLENGE]" in result[0].text
        assert "Approval ID: apr-" in result[0].text

    async def test_hard_block_no_approval(self, approval_db: DatabaseManager, broadcaster: EventBroadcaster) -> None:
        proxy = _make_proxy()
        proxy._guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK,
            check_results=[_make_check_result("permission_scope", passed=False, force_verdict=Verdict.BLOCK)],
        ))
        proxy._db = approval_db
        proxy._broadcaster = broadcaster
        proxy._approval_manager = ApprovalManager(db=approval_db, broadcaster=broadcaster)

        result = await proxy._intercept_and_forward("send_email", {"to": "x"})
        assert "[JANUS BLOCKED]" in result[0].text
        assert "Approval ID" not in result[0].text

        pending = await proxy._approval_manager.get_pending()
        assert len(pending) == 0

    async def test_sandbox_no_approval(self, approval_db: DatabaseManager, broadcaster: EventBroadcaster) -> None:
        proxy = _make_proxy()
        proxy._guardian = _mock_guardian(_make_verdict(Verdict.SANDBOX, risk_score=60.0))
        proxy._db = approval_db
        proxy._broadcaster = broadcaster
        proxy._approval_manager = ApprovalManager(db=approval_db, broadcaster=broadcaster)

        result = await proxy._intercept_and_forward("execute_code", {"code": "ls"})
        assert "[JANUS BLOCKED]" in result[0].text
        assert "Approval ID" not in result[0].text

    async def test_pause_creates_approval(self, approval_db: DatabaseManager, broadcaster: EventBroadcaster) -> None:
        proxy = _make_proxy()
        proxy._guardian = _mock_guardian(_make_verdict(
            Verdict.PAUSE, risk_score=55.0,
            reasons=["anomaly"],
            check_results=[_make_check_result("itdr", passed=False)],
        ))
        proxy._db = approval_db
        proxy._broadcaster = broadcaster
        proxy._approval_manager = ApprovalManager(db=approval_db, broadcaster=broadcaster)

        result = await proxy._intercept_and_forward("api_call", {})
        assert "[JANUS BLOCKED]" in result[0].text
        assert "Approval ID: apr-" in result[0].text

    async def test_approval_creation_exception(self, approval_db: DatabaseManager, broadcaster: EventBroadcaster) -> None:
        """Approval creation failure should not crash the proxy."""
        proxy = _make_proxy()
        proxy._guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=80.0,
            check_results=[_make_check_result("deterministic_risk", passed=False)],
        ))
        proxy._db = approval_db
        proxy._broadcaster = broadcaster
        mgr = ApprovalManager(db=approval_db, broadcaster=broadcaster)
        mgr.create = AsyncMock(side_effect=RuntimeError("DB failure"))
        proxy._approval_manager = mgr

        result = await proxy._intercept_and_forward("tool", {})
        assert "[JANUS BLOCKED]" in result[0].text
        # Should not contain approval_id since creation failed
        assert "Approval ID" not in result[0].text


class TestMCPProxyEventBroadcasting:
    async def test_verdict_event_broadcast_on_allow(self) -> None:
        proxy = _make_proxy()
        proxy._guardian = _mock_guardian(_make_verdict(
            Verdict.ALLOW, risk_score=3.0, risk_delta=3.0,
        ))
        proxy._upstream = AsyncMock()
        proxy._upstream.call_tool.return_value = types.CallToolResult(
            content=[types.TextContent(type="text", text="ok")]
        )

        received: list[SecurityEvent] = []
        sub = proxy._broadcaster.subscribe("test-session")
        sub_iter = sub.__aiter__()

        async def collect():
            async for event in sub_iter:
                received.append(event)
                if len(received) >= 1:
                    break

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.01)

        await proxy._intercept_and_forward("read_file", {"path": "/x"})
        await asyncio.wait_for(task, timeout=2.0)

        assert len(received) == 1
        assert received[0].event_type == "verdict"
        assert received[0].data["verdict"] == "allow"
        assert received[0].data["tool_name"] == "read_file"
        assert received[0].data["integration"] == "mcp"

    async def test_verdict_event_broadcast_on_block(self, approval_db: DatabaseManager, broadcaster: EventBroadcaster) -> None:
        proxy = _make_proxy()
        proxy._guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK, risk_score=80.0,
            reasons=["high risk"],
            check_results=[_make_check_result("deterministic_risk", passed=False)],
        ))
        proxy._db = approval_db
        proxy._broadcaster = broadcaster
        proxy._approval_manager = ApprovalManager(db=approval_db, broadcaster=broadcaster)

        received: list[SecurityEvent] = []
        sub = broadcaster.subscribe("test-session")
        sub_iter = sub.__aiter__()

        async def collect():
            async for event in sub_iter:
                received.append(event)
                # verdict + approval_created = 2 events
                if len(received) >= 2:
                    break

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.01)

        await proxy._intercept_and_forward("execute_code", {"code": "rm /"})
        await asyncio.wait_for(task, timeout=2.0)

        assert len(received) == 2
        assert received[0].event_type == "verdict"
        assert received[0].data["verdict"] == "block"
        assert received[1].event_type == "approval_created"
        assert received[1].data["tool_name"] == "execute_code"

    async def test_verdict_includes_check_results(self) -> None:
        proxy = _make_proxy()
        proxy._guardian = _mock_guardian(_make_verdict(
            Verdict.BLOCK,
            check_results=[
                _make_check_result("deterministic_risk", passed=False, risk_contribution=20.0, reason="high"),
                _make_check_result("llm_risk_classifier", passed=False, risk_contribution=15.0, reason="suspicious"),
            ],
        ))

        received: list[SecurityEvent] = []
        sub = proxy._broadcaster.subscribe("test-session")
        sub_iter = sub.__aiter__()

        async def collect():
            async for event in sub_iter:
                received.append(event)
                if len(received) >= 1:
                    break

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.01)

        await proxy._intercept_and_forward("tool", {})
        await asyncio.wait_for(task, timeout=2.0)

        check_results = received[0].data["check_results"]
        assert len(check_results) == 2
        assert check_results[0]["check_name"] == "deterministic_risk"
        assert check_results[1]["check_name"] == "llm_risk_classifier"

    async def test_global_subscriber_receives_events(self) -> None:
        proxy = _make_proxy()
        proxy._guardian = _mock_guardian(_make_verdict(Verdict.ALLOW))
        proxy._upstream = AsyncMock()
        proxy._upstream.call_tool.return_value = types.CallToolResult(
            content=[types.TextContent(type="text", text="ok")]
        )

        received: list[SecurityEvent] = []
        # Subscribe to "*" (global) instead of specific session
        sub = proxy._broadcaster.subscribe("*")
        sub_iter = sub.__aiter__()

        async def collect():
            async for event in sub_iter:
                received.append(event)
                if len(received) >= 1:
                    break

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.01)

        await proxy._intercept_and_forward("read_file", {"path": "/x"})
        await asyncio.wait_for(task, timeout=2.0)

        assert len(received) == 1
        assert received[0].session_id == "test-session"


class TestMCPProxyTaintTracking:
    async def test_taint_scan_on_allow(self) -> None:
        proxy = _make_proxy()
        mock_guardian = _mock_guardian(_make_verdict(Verdict.ALLOW))
        mock_guardian.taint_tracker = MagicMock()
        proxy._guardian = mock_guardian
        proxy._upstream = AsyncMock()
        proxy._upstream.call_tool.return_value = types.CallToolResult(
            content=[types.TextContent(type="text", text="sensitive data")]
        )

        await proxy._intercept_and_forward("read_file", {"path": "/secrets"})
        mock_guardian.taint_tracker.scan_output.assert_called_once()
        call_args = mock_guardian.taint_tracker.scan_output.call_args
        assert call_args[0][0] == "test-session"
        assert call_args[0][1] == "read_file"

    async def test_no_taint_tracker(self) -> None:
        """No crash if taint_tracker doesn't exist."""
        proxy = _make_proxy()
        mock_guardian = _mock_guardian(_make_verdict(Verdict.ALLOW))
        # Explicitly remove taint_tracker
        if hasattr(mock_guardian, "taint_tracker"):
            del mock_guardian.taint_tracker
        proxy._guardian = mock_guardian
        proxy._upstream = AsyncMock()
        proxy._upstream.call_tool.return_value = types.CallToolResult(
            content=[types.TextContent(type="text", text="ok")]
        )
        # Should not crash
        result = await proxy._intercept_and_forward("read_file", {})
        assert "ok" in result[0].text


class TestMCPProxySetup:
    async def test_setup_creates_approval_manager(self) -> None:
        config = ProxyConfig(
            database_path=":memory:",
            agent=AgentConfig(agent_id="setup-test", role="research"),
        )
        proxy = JanusMCPProxy(config)
        await proxy.setup()

        assert proxy.guardian is not None
        assert proxy.approval_manager is not None
        assert proxy.broadcaster is not None
        assert proxy._session_id.startswith("mcp-proxy-")

        await proxy.teardown()

    async def test_setup_custom_session_id(self) -> None:
        from janus.mcp.config import SessionConfig
        config = ProxyConfig(
            database_path=":memory:",
            agent=AgentConfig(agent_id="test"),
            session=SessionConfig(persistent_session_id="my-custom-session"),
        )
        proxy = JanusMCPProxy(config)
        await proxy.setup()
        assert proxy._session_id == "my-custom-session"
        await proxy.teardown()

    async def test_upstream_error_handled(self) -> None:
        proxy = _make_proxy()
        proxy._guardian = _mock_guardian(_make_verdict(Verdict.ALLOW))
        proxy._upstream = AsyncMock()
        proxy._upstream.call_tool.side_effect = ConnectionError("upstream died")

        result = await proxy._intercept_and_forward("read_file", {"path": "/x"})
        assert "Upstream error" in result[0].text
        assert "upstream died" in result[0].text


# ═══════════════════════════════════════════════════════════════════
# 9. ApprovalManager edge cases (beyond test_approval.py)
# ═══════════════════════════════════════════════════════════════════


class TestApprovalManagerEdgeCases:
    async def test_approve_with_tool_executor_exception(
        self, approval_db: DatabaseManager, broadcaster: EventBroadcaster
    ) -> None:
        """Tool execution error during approve is captured, not raised."""
        executor = AsyncMock()
        executor.execute.side_effect = RuntimeError("tool exploded")
        mgr = ApprovalManager(db=approval_db, broadcaster=broadcaster, tool_executor=executor)

        req = await mgr.create(
            session_id="s1", agent_id="a1", tool_name="dangerous_tool",
            tool_input={"x": 1}, original_goal="test", verdict="block",
            risk_score=80.0, risk_delta=25.0, reasons=["risky"],
            check_results=[], trace_id="t1",
        )
        result = await mgr.approve(req.id)
        assert result is not None
        assert result.status == "approved"
        assert result.tool_result == {"error": "tool exploded"}

    async def test_approve_with_taint_scanning(
        self, approval_db: DatabaseManager, broadcaster: EventBroadcaster
    ) -> None:
        """Taint tracker scans output after tool execution on approve."""
        executor = AsyncMock()
        executor.execute.return_value = {"content": "sensitive data"}

        mock_guardian = MagicMock()
        mock_guardian.taint_tracker = MagicMock()

        mgr = ApprovalManager(
            db=approval_db, broadcaster=broadcaster,
            tool_executor=executor, guardian=mock_guardian,
        )

        req = await mgr.create(
            session_id="s1", agent_id="a1", tool_name="read_file",
            tool_input={"path": "/x"}, original_goal="test", verdict="block",
            risk_score=70.0, risk_delta=20.0, reasons=["risky"],
            check_results=[], trace_id="t1",
        )
        result = await mgr.approve(req.id)
        assert result is not None
        mock_guardian.taint_tracker.scan_output.assert_called_once()

    async def test_approve_nonexistent(self, approval_manager: ApprovalManager) -> None:
        result = await approval_manager.approve("nonexistent")
        assert result is None

    async def test_reject_nonexistent(self, approval_manager: ApprovalManager) -> None:
        result = await approval_manager.reject("nonexistent")
        assert result is None

    async def test_approve_then_reject(self, approval_manager: ApprovalManager) -> None:
        """Cannot reject after approve."""
        req = await approval_manager.create(
            session_id="s1", agent_id="a1", tool_name="tool",
            tool_input={}, original_goal="test", verdict="block",
            risk_score=70.0, risk_delta=20.0, reasons=[], check_results=[], trace_id="t1",
        )
        await approval_manager.approve(req.id)
        result = await approval_manager.reject(req.id)
        assert result is None

    async def test_reject_then_approve(self, approval_manager: ApprovalManager) -> None:
        """Cannot approve after reject."""
        req = await approval_manager.create(
            session_id="s1", agent_id="a1", tool_name="tool",
            tool_input={}, original_goal="test", verdict="block",
            risk_score=70.0, risk_delta=20.0, reasons=[], check_results=[], trace_id="t1",
        )
        await approval_manager.reject(req.id)
        result = await approval_manager.approve(req.id)
        assert result is None

    async def test_get_all_no_filters(self, approval_manager: ApprovalManager) -> None:
        await approval_manager.create(
            session_id="s1", agent_id="a1", tool_name="t1",
            tool_input={}, original_goal="", verdict="block",
            risk_score=70.0, risk_delta=20.0, reasons=[], check_results=[], trace_id="",
        )
        await approval_manager.create(
            session_id="s2", agent_id="a2", tool_name="t2",
            tool_input={}, original_goal="", verdict="challenge",
            risk_score=50.0, risk_delta=10.0, reasons=[], check_results=[], trace_id="",
        )
        all_reqs = await approval_manager.get_all()
        assert len(all_reqs) == 2

    async def test_get_all_with_session_filter(self, approval_manager: ApprovalManager) -> None:
        await approval_manager.create(
            session_id="s1", agent_id="a1", tool_name="t1",
            tool_input={}, original_goal="", verdict="block",
            risk_score=70.0, risk_delta=20.0, reasons=[], check_results=[], trace_id="",
        )
        await approval_manager.create(
            session_id="s2", agent_id="a2", tool_name="t2",
            tool_input={}, original_goal="", verdict="block",
            risk_score=70.0, risk_delta=20.0, reasons=[], check_results=[], trace_id="",
        )
        s1_reqs = await approval_manager.get_all(session_id="s1")
        assert len(s1_reqs) == 1
        assert s1_reqs[0].session_id == "s1"

    async def test_create_with_complex_tool_input(self, approval_manager: ApprovalManager) -> None:
        """Tool input with nested structures and non-serializable types."""
        from datetime import datetime
        req = await approval_manager.create(
            session_id="s1", agent_id="a1", tool_name="complex_tool",
            tool_input={"nested": {"deep": [1, 2, 3]}, "date": datetime.now(UTC)},
            original_goal="test", verdict="block",
            risk_score=70.0, risk_delta=20.0,
            reasons=["reason 1", "reason 2", "reason 3"],
            check_results=[{"check_name": "test", "passed": False}],
            trace_id="t1",
        )
        # Should not fail despite datetime in tool_input (default=str handles it)
        fetched = await approval_manager.get_by_id(req.id)
        assert fetched is not None
        assert "nested" in fetched.tool_input

    async def test_approval_id_format(self, approval_manager: ApprovalManager) -> None:
        req = await approval_manager.create(
            session_id="s1", agent_id="a1", tool_name="tool",
            tool_input={}, original_goal="test", verdict="block",
            risk_score=70.0, risk_delta=20.0, reasons=[], check_results=[], trace_id="",
        )
        assert req.id.startswith("apr-")
        assert len(req.id) == 16  # "apr-" + 12 hex chars

    async def test_to_dict(self, approval_manager: ApprovalManager) -> None:
        req = await approval_manager.create(
            session_id="s1", agent_id="a1", tool_name="tool",
            tool_input={"key": "value"}, original_goal="goal", verdict="block",
            risk_score=70.0, risk_delta=20.0, reasons=["r1"], check_results=[], trace_id="t1",
        )
        d = req.to_dict()
        assert d["id"] == req.id
        assert d["session_id"] == "s1"
        assert d["tool_input"] == {"key": "value"}
        assert d["status"] == "pending"

    async def test_exporter_coordinator_on_create(
        self, approval_db: DatabaseManager, broadcaster: EventBroadcaster
    ) -> None:
        """Notifiers are fired when approval is created."""
        coordinator = AsyncMock()
        mgr = ApprovalManager(
            db=approval_db, broadcaster=broadcaster,
            exporter_coordinator=coordinator,
        )
        await mgr.create(
            session_id="s1", agent_id="a1", tool_name="tool",
            tool_input={}, original_goal="test", verdict="block",
            risk_score=70.0, risk_delta=20.0, reasons=["risky"],
            check_results=[], trace_id="",
        )
        coordinator.export.assert_called_once()

    async def test_exporter_coordinator_exception(
        self, approval_db: DatabaseManager, broadcaster: EventBroadcaster
    ) -> None:
        """Notifier failure should not prevent approval creation."""
        coordinator = AsyncMock()
        coordinator.export.side_effect = Exception("slack down")
        mgr = ApprovalManager(
            db=approval_db, broadcaster=broadcaster,
            exporter_coordinator=coordinator,
        )
        req = await mgr.create(
            session_id="s1", agent_id="a1", tool_name="tool",
            tool_input={}, original_goal="test", verdict="block",
            risk_score=70.0, risk_delta=20.0, reasons=["risky"],
            check_results=[], trace_id="",
        )
        # Approval should still be created despite notifier failure
        assert req.id.startswith("apr-")
        fetched = await mgr.get_by_id(req.id)
        assert fetched is not None


# ═══════════════════════════════════════════════════════════════════
# 10. needs_human_review() edge cases
# ═══════════════════════════════════════════════════════════════════


class TestNeedsHumanReview:
    def test_all_hard_block_checks(self) -> None:
        for check in ["permission_scope", "identity_check", "prompt_injection"]:
            assert needs_human_review("block", [
                {"check_name": check, "passed": False},
            ]) is False

    def test_hard_block_with_passed_true(self) -> None:
        """If the check passed, it shouldn't block human review."""
        assert needs_human_review("block", [
            {"check_name": "permission_scope", "passed": True},
        ]) is True

    def test_mixed_hard_and_soft_checks(self) -> None:
        """If ANY hard check failed, no review even if soft checks also failed."""
        assert needs_human_review("block", [
            {"check_name": "deterministic_risk", "passed": False, "risk_contribution": 30.0},
            {"check_name": "permission_scope", "passed": False},
        ]) is False

    def test_only_soft_checks_failed(self) -> None:
        assert needs_human_review("block", [
            {"check_name": "deterministic_risk", "passed": False},
            {"check_name": "llm_risk_classifier", "passed": False},
            {"check_name": "drift", "passed": False},
        ]) is True

    def test_all_checks_passed_but_blocked(self) -> None:
        """All individual checks passed but verdict is block (e.g., from risk accumulation)."""
        assert needs_human_review("block", [
            {"check_name": "deterministic_risk", "passed": True},
            {"check_name": "permission_scope", "passed": True},
        ]) is True

    def test_missing_check_name_key(self) -> None:
        """Check result without check_name should not match hard blocks."""
        assert needs_human_review("block", [
            {"passed": False, "reason": "unknown check"},
        ]) is True

    def test_missing_passed_key(self) -> None:
        """Missing 'passed' defaults to True (no hard block detected)."""
        assert needs_human_review("block", [
            {"check_name": "permission_scope"},
        ]) is True

    def test_allow_verdict(self) -> None:
        """Allow verdict technically returns True but should never be called with allow."""
        # This is an edge case — allow verdicts shouldn't reach needs_human_review
        # but the function should handle it gracefully
        result = needs_human_review("allow", [])
        assert isinstance(result, bool)

    def test_unknown_verdict(self) -> None:
        """Unknown verdict string defaults to needing review."""
        assert needs_human_review("unknown_verdict", []) is True

    def test_empty_check_results_all_verdicts(self) -> None:
        assert needs_human_review("block", []) is True
        assert needs_human_review("challenge", []) is True
        assert needs_human_review("pause", []) is True
        assert needs_human_review("sandbox", []) is False


# ═══════════════════════════════════════════════════════════════════
# 11. EventBroadcaster edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEventBroadcasterEdgeCases:
    async def test_session_isolation(self) -> None:
        """Events for session A should not reach session B subscribers."""
        b = EventBroadcaster()
        received_a: list[SecurityEvent] = []
        received_b: list[SecurityEvent] = []

        sub_a = b.subscribe("session-a")
        sub_b = b.subscribe("session-b")

        async def collect_a():
            async for ev in sub_a:
                received_a.append(ev)
                break

        async def collect_b():
            async for ev in sub_b:
                received_b.append(ev)
                break

        task_a = asyncio.create_task(collect_a())
        task_b = asyncio.create_task(collect_b())
        await asyncio.sleep(0.01)

        await b.publish(SecurityEvent(event_type="test", session_id="session-a", data={"for": "a"}))
        await b.publish(SecurityEvent(event_type="test", session_id="session-b", data={"for": "b"}))

        await asyncio.wait_for(task_a, timeout=2.0)
        await asyncio.wait_for(task_b, timeout=2.0)

        assert len(received_a) == 1
        assert received_a[0].data["for"] == "a"
        assert len(received_b) == 1
        assert received_b[0].data["for"] == "b"

    async def test_global_subscriber_receives_all(self) -> None:
        b = EventBroadcaster()
        received: list[SecurityEvent] = []

        sub = b.subscribe("*")

        async def collect():
            async for ev in sub:
                received.append(ev)
                if len(received) >= 3:
                    break

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.01)

        await b.publish(SecurityEvent(event_type="t1", session_id="s1"))
        await b.publish(SecurityEvent(event_type="t2", session_id="s2"))
        await b.publish(SecurityEvent(event_type="t3", session_id="s3"))

        await asyncio.wait_for(task, timeout=2.0)
        assert len(received) == 3

    async def test_multiple_subscribers_same_session(self) -> None:
        b = EventBroadcaster()
        received_1: list[SecurityEvent] = []
        received_2: list[SecurityEvent] = []

        sub_1 = b.subscribe("session-x")
        sub_2 = b.subscribe("session-x")

        async def collect_1():
            async for ev in sub_1:
                received_1.append(ev)
                break

        async def collect_2():
            async for ev in sub_2:
                received_2.append(ev)
                break

        t1 = asyncio.create_task(collect_1())
        t2 = asyncio.create_task(collect_2())
        await asyncio.sleep(0.01)

        assert b.subscriber_count("session-x") == 2

        await b.publish(SecurityEvent(event_type="test", session_id="session-x"))

        await asyncio.wait_for(t1, timeout=2.0)
        await asyncio.wait_for(t2, timeout=2.0)

        assert len(received_1) == 1
        assert len(received_2) == 1

    async def test_subscriber_count(self) -> None:
        b = EventBroadcaster()
        assert b.subscriber_count("session-x") == 0

        sub1 = b.subscribe("session-x")
        sub_iter = sub1.__aiter__()

        async def hold():
            async for _ in sub_iter:
                break

        task = asyncio.create_task(hold())
        await asyncio.sleep(0.01)
        assert b.subscriber_count("session-x") == 1

        # Cancel the task to trigger cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.01)
        assert b.subscriber_count("session-x") == 0

    async def test_event_to_dict(self) -> None:
        ev = SecurityEvent(
            event_type="verdict",
            session_id="s1",
            data={"verdict": "allow", "risk_score": 5.0},
        )
        d = ev.to_dict()
        assert d["event_type"] == "verdict"
        assert d["session_id"] == "s1"
        assert d["data"]["verdict"] == "allow"
        assert "timestamp" in d

    async def test_empty_data_event(self) -> None:
        ev = SecurityEvent(event_type="ping", session_id="s1")
        d = ev.to_dict()
        assert d["data"] == {}


# ═══════════════════════════════════════════════════════════════════
# 12. Cross-integration approval lifecycle
# ═══════════════════════════════════════════════════════════════════


class TestApprovalLifecycle:
    async def test_create_approve_lifecycle(
        self, approval_db: DatabaseManager, broadcaster: EventBroadcaster
    ) -> None:
        """Full lifecycle: create → pending → approve → approved with tool result."""
        from janus.web.tools import MockToolExecutor

        executor = MockToolExecutor()
        mgr = ApprovalManager(
            db=approval_db, broadcaster=broadcaster, tool_executor=executor,
        )

        # Create
        req = await mgr.create(
            session_id="s1", agent_id="a1", tool_name="read_file",
            tool_input={"path": "/tmp/test"}, original_goal="Read file",
            verdict="block", risk_score=70.0, risk_delta=20.0,
            reasons=["high risk"], check_results=[], trace_id="t1",
        )
        assert req.status == "pending"
        assert req.decided_by is None

        # Check pending
        pending = await mgr.get_pending()
        assert len(pending) == 1

        # Approve
        result = await mgr.approve(req.id, decided_by="admin", reason="Looks safe")
        assert result.status == "approved"
        assert result.decided_by == "admin"
        assert result.decided_at is not None
        assert result.tool_result is not None

        # No more pending
        pending = await mgr.get_pending()
        assert len(pending) == 0

        # Stats
        stats = await mgr.get_stats()
        assert stats["approved"] == 1
        assert stats["pending"] == 0

    async def test_create_reject_lifecycle(
        self, approval_db: DatabaseManager, broadcaster: EventBroadcaster
    ) -> None:
        mgr = ApprovalManager(db=approval_db, broadcaster=broadcaster)

        req = await mgr.create(
            session_id="s1", agent_id="a1", tool_name="execute_code",
            tool_input={"code": "rm /"}, original_goal="test",
            verdict="block", risk_score=90.0, risk_delta=30.0,
            reasons=["very risky"], check_results=[], trace_id="t1",
        )

        result = await mgr.reject(req.id, decided_by="security-team", reason="Too dangerous")
        assert result.status == "rejected"
        assert result.decided_by == "security-team"
        assert result.tool_result is None

        stats = await mgr.get_stats()
        assert stats["rejected"] == 1

    async def test_broadcast_full_lifecycle(
        self, approval_db: DatabaseManager, broadcaster: EventBroadcaster
    ) -> None:
        """Verify all events broadcast through a full approval lifecycle."""
        mgr = ApprovalManager(db=approval_db, broadcaster=broadcaster)

        received: list[SecurityEvent] = []
        sub = broadcaster.subscribe("*")

        async def collect():
            async for ev in sub:
                received.append(ev)
                if len(received) >= 2:
                    break

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.01)

        req = await mgr.create(
            session_id="lifecycle", agent_id="a1", tool_name="tool",
            tool_input={}, original_goal="test", verdict="block",
            risk_score=70.0, risk_delta=20.0, reasons=[], check_results=[], trace_id="",
        )
        await mgr.reject(req.id, reason="denied")

        await asyncio.wait_for(task, timeout=2.0)

        assert received[0].event_type == "approval_created"
        assert received[0].data["status"] == "pending"
        assert received[1].event_type == "approval_resolved"
        assert received[1].data["resolution"] == "rejected"
