"""Tests for the FastAPI application."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from sentinel.web.app import _setup, _teardown, create_app, state


@pytest.fixture
async def client():
    app = create_app()
    await _setup()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        await _teardown()
        # Reset module-level state between tests
        state.guardian = None
        state.registry = None
        state.risk_engine = None
        state.session_store = None
        state.db = None
        state.chat_agents.clear()
        state.sessions.clear()


async def test_health_endpoint(client: AsyncClient) -> None:
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


async def test_list_sessions_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_agents(client: AsyncClient) -> None:
    resp = await client.get("/api/agents")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["agent_id"] == "demo-agent"


async def test_create_session(client: AsyncClient) -> None:
    resp = await client.post("/api/sessions", json={
        "agent_id": "demo-agent",
        "original_goal": "Research public API docs",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["agent_id"] == "demo-agent"
    assert data["risk_score"] == 0.0
