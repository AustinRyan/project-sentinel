"""Tests for the production monitoring dashboard backend: check_results propagation,
new endpoints (/events, /taint, /health/full), and global WebSocket fan-out."""
from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from janus.core.decision import CheckResult, SecurityVerdict, Verdict
from janus.web.app import _setup, _teardown, create_app, state
from janus.web.events import EventBroadcaster, SecurityEvent

# ── check_results propagation ────────────────────────────────────────

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
        state.chat_agents.clear()
        state.sessions.clear()
        from janus.tier import current_tier
        current_tier.reset()


async def test_verdict_contains_check_results(client: AsyncClient) -> None:
    """wrap_tool_call should propagate check_results from the pipeline."""
    assert state.guardian is not None

    verdict = await state.guardian.wrap_tool_call(
        agent_id="demo-agent",
        session_id="monitor-test-1",
        original_goal="Test monitoring",
        tool_name="read_file",
        tool_input={"path": "/etc/hosts"},
    )

    assert isinstance(verdict, SecurityVerdict)
    assert len(verdict.check_results) > 0
    assert all(isinstance(cr, CheckResult) for cr in verdict.check_results)
    # At minimum we should see identity_check and permission_scope
    names = [cr.check_name for cr in verdict.check_results]
    assert "identity_check" in names
    assert "permission_scope" in names


async def test_check_results_have_correct_structure(client: AsyncClient) -> None:
    """Each CheckResult should have the expected fields."""
    assert state.guardian is not None

    verdict = await state.guardian.wrap_tool_call(
        agent_id="demo-agent",
        session_id="monitor-test-2",
        original_goal="Test structure",
        tool_name="execute_code",
        tool_input={"code": "print('hello')"},
    )

    for cr in verdict.check_results:
        assert isinstance(cr.check_name, str)
        assert isinstance(cr.passed, bool)
        assert isinstance(cr.risk_contribution, float)
        assert isinstance(cr.reason, str)
        assert isinstance(cr.metadata, dict)


async def test_blocked_verdict_still_has_check_results(client: AsyncClient) -> None:
    """Even short-circuited (blocked) verdicts should contain check_results."""
    assert state.guardian is not None

    # Unregistered agent should be blocked at identity check
    verdict = await state.guardian.wrap_tool_call(
        agent_id="nonexistent-agent",
        session_id="monitor-test-3",
        original_goal="Test block",
        tool_name="read_file",
        tool_input={},
    )

    assert verdict.verdict == Verdict.BLOCK
    assert len(verdict.check_results) > 0


# ── New API endpoints ────────────────────────────────────────────────

async def test_session_events_endpoint(client: AsyncClient) -> None:
    """GET /api/sessions/{id}/events should return risk event history."""
    assert state.guardian is not None

    # Create a session and evaluate a tool call to generate events
    await state.guardian.wrap_tool_call(
        agent_id="demo-agent",
        session_id="events-test-1",
        original_goal="Generate events",
        tool_name="execute_code",
        tool_input={"code": "import os"},
    )

    resp = await client.get("/api/sessions/events-test-1/events")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    event = data[0]
    assert "risk_delta" in event
    assert "new_score" in event
    assert "tool_name" in event
    assert "timestamp" in event


async def test_session_events_empty(client: AsyncClient) -> None:
    """Events endpoint for unknown session should return empty list."""
    resp = await client.get("/api/sessions/nonexistent/events")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_session_taint_endpoint(client: AsyncClient) -> None:
    """GET /api/sessions/{id}/taint should return taint entries."""
    resp = await client.get("/api/sessions/taint-test-1/taint")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


async def test_health_full_endpoint(client: AsyncClient) -> None:
    """GET /api/health/full should return full health metrics."""
    resp = await client.get("/api/health/full")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "total_requests" in data
    assert "successful_requests" in data
    assert "failed_requests" in data
    assert "avg_latency_ms" in data
    assert "p95_latency_ms" in data
    assert "error_rate" in data
    assert "circuit_breaker" in data
    assert "active_sessions" in data


async def test_health_full_reflects_requests(client: AsyncClient) -> None:
    """Health metrics should update after evaluating tool calls."""
    assert state.guardian is not None

    await state.guardian.wrap_tool_call(
        agent_id="demo-agent",
        session_id="health-test-1",
        original_goal="Test health",
        tool_name="read_file",
        tool_input={},
    )

    resp = await client.get("/api/health/full")
    data = resp.json()
    assert data["total_requests"] >= 1
    assert data["successful_requests"] >= 1


# ── Evaluate endpoint includes check_results in broadcast ────────────

async def test_evaluate_broadcasts_check_results(client: AsyncClient) -> None:
    """POST /evaluate should include check_results in the WS broadcast."""
    received: list[SecurityEvent] = []

    async def listener():
        async for event in state.broadcaster.subscribe("eval-cr-test"):
            received.append(event)
            break

    task = asyncio.create_task(listener())
    await asyncio.sleep(0.05)

    resp = await client.post("/api/evaluate", json={
        "agent_id": "demo-agent",
        "session_id": "eval-cr-test",
        "tool_name": "read_file",
        "tool_input": {"path": "/tmp/test"},
        "original_goal": "Test eval broadcast",
    })
    assert resp.status_code == 200

    await asyncio.wait_for(task, timeout=3.0)
    assert len(received) == 1
    assert "check_results" in received[0].data
    assert isinstance(received[0].data["check_results"], list)
    assert len(received[0].data["check_results"]) > 0


# ── Global WebSocket fan-out ─────────────────────────────────────────

async def test_global_fanout_receives_all_sessions() -> None:
    """Subscribing to '*' should receive events from any session."""
    broadcaster = EventBroadcaster()
    received: list[SecurityEvent] = []

    async def listener():
        async for event in broadcaster.subscribe("*"):
            received.append(event)
            if len(received) >= 2:
                break

    task = asyncio.create_task(listener())
    await asyncio.sleep(0.05)

    await broadcaster.publish(SecurityEvent(
        event_type="verdict",
        session_id="session-A",
        data={"verdict": "allow"},
    ))
    await broadcaster.publish(SecurityEvent(
        event_type="verdict",
        session_id="session-B",
        data={"verdict": "block"},
    ))

    await asyncio.wait_for(task, timeout=2.0)
    assert len(received) == 2
    assert received[0].session_id == "session-A"
    assert received[1].session_id == "session-B"


async def test_global_fanout_does_not_duplicate_for_star_session() -> None:
    """Events with session_id='*' should NOT double-deliver to '*' subscribers."""
    broadcaster = EventBroadcaster()
    received: list[SecurityEvent] = []

    async def listener():
        async for event in broadcaster.subscribe("*"):
            received.append(event)
            break

    task = asyncio.create_task(listener())
    await asyncio.sleep(0.05)

    await broadcaster.publish(SecurityEvent(
        event_type="test",
        session_id="*",
        data={},
    ))

    await asyncio.wait_for(task, timeout=2.0)
    # Should receive exactly 1, not 2
    assert len(received) == 1


async def test_session_subscriber_still_works_with_global() -> None:
    """Session-specific subscribers should still work alongside global."""
    broadcaster = EventBroadcaster()
    session_received: list[SecurityEvent] = []
    global_received: list[SecurityEvent] = []

    async def session_listener():
        async for event in broadcaster.subscribe("session-X"):
            session_received.append(event)
            break

    async def global_listener():
        async for event in broadcaster.subscribe("*"):
            global_received.append(event)
            break

    t1 = asyncio.create_task(session_listener())
    t2 = asyncio.create_task(global_listener())
    await asyncio.sleep(0.05)

    await broadcaster.publish(SecurityEvent(
        event_type="verdict",
        session_id="session-X",
        data={"verdict": "allow"},
    ))

    await asyncio.wait_for(t1, timeout=2.0)
    await asyncio.wait_for(t2, timeout=2.0)

    assert len(session_received) == 1
    assert len(global_received) == 1
