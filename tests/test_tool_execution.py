"""Tests for the real tool execution system — registry, executor, webhook, API routes."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from janus.storage.database import DatabaseManager
from janus.tools.executor import ToolExecutor, WebhookExecutor, _resolve_credential
from janus.tools.models import RegisteredTool
from janus.tools.registry import ToolRegistry
from janus.web.app import _setup, _teardown, create_app, state

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
async def db():
    database = DatabaseManager(":memory:")
    await database.connect()
    await database.apply_migrations()
    yield database
    await database.close()


@pytest.fixture
async def registry(db: DatabaseManager) -> ToolRegistry:
    return ToolRegistry(db)


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setenv("JANUS_DB_PATH", ":memory:")
    monkeypatch.setenv("JANUS_DEV_MODE", "true")
    monkeypatch.setenv("JANUS_MOCK_TOOLS", "false")
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
        state.tool_registry = None
        state.tool_executor = None
        state.chat_agents.clear()
        state.sessions.clear()
        from janus.tier import current_tier
        current_tier.reset()


@pytest.fixture
async def mock_client(monkeypatch):
    """Client with JANUS_MOCK_TOOLS=true for mock mode testing."""
    monkeypatch.setenv("JANUS_DB_PATH", ":memory:")
    monkeypatch.setenv("JANUS_DEV_MODE", "true")
    monkeypatch.setenv("JANUS_MOCK_TOOLS", "true")
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
        state.tool_registry = None
        state.tool_executor = None
        state.chat_agents.clear()
        state.sessions.clear()
        from janus.tier import current_tier
        current_tier.reset()


# ── RegisteredTool model tests ─────────────────────────────────────────


class TestRegisteredTool:
    def test_new_id_format(self):
        tid = RegisteredTool.new_id()
        assert tid.startswith("tool-")
        assert len(tid) == 17  # "tool-" + 12 hex chars

    def test_to_claude_tool(self):
        tool = RegisteredTool(
            id="tool-abc",
            name="search_db",
            description="Search the database",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        ct = tool.to_claude_tool()
        assert ct["name"] == "search_db"
        assert ct["description"] == "Search the database"
        assert "properties" in ct["input_schema"]

    def test_to_dict(self):
        tool = RegisteredTool(id="tool-abc", name="test_tool")
        d = tool.to_dict()
        assert d["id"] == "tool-abc"
        assert d["name"] == "test_tool"
        assert d["type"] == "webhook"
        assert d["is_active"] is True

    def test_timestamps_auto_set(self):
        tool = RegisteredTool(id="tool-abc", name="test_tool")
        assert tool.created_at != ""
        assert tool.updated_at != ""


# ── ToolRegistry tests ─────────────────────────────────────────────────


class TestToolRegistry:
    async def test_register_tool(self, registry: ToolRegistry):
        tool = await registry.register(
            name="my_api",
            description="Call my API",
            type="webhook",
            endpoint="https://api.example.com/tools/my_api",
            auth_type="bearer",
            auth_credential="$MY_TOKEN",
        )
        assert tool.name == "my_api"
        assert tool.type == "webhook"
        assert tool.endpoint == "https://api.example.com/tools/my_api"
        assert tool.auth_type == "bearer"
        assert tool.id.startswith("tool-")

    async def test_register_duplicate_fails(self, registry: ToolRegistry):
        await registry.register(name="dup_tool", endpoint="https://example.com")
        with pytest.raises(ValueError, match="already registered"):
            await registry.register(name="dup_tool", endpoint="https://other.com")

    async def test_get_by_name(self, registry: ToolRegistry):
        await registry.register(name="find_me", endpoint="https://example.com")
        tool = await registry.get_by_name("find_me")
        assert tool is not None
        assert tool.name == "find_me"

    async def test_get_by_name_not_found(self, registry: ToolRegistry):
        tool = await registry.get_by_name("nonexistent")
        assert tool is None

    async def test_get_by_id(self, registry: ToolRegistry):
        created = await registry.register(name="by_id_tool", endpoint="https://example.com")
        found = await registry.get_by_id(created.id)
        assert found is not None
        assert found.name == "by_id_tool"

    async def test_list_tools(self, registry: ToolRegistry):
        await registry.register(name="tool_a", endpoint="https://a.com")
        await registry.register(name="tool_b", endpoint="https://b.com")
        tools = await registry.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"tool_a", "tool_b"}

    async def test_list_tools_active_only(self, registry: ToolRegistry):
        tool = await registry.register(name="deactivated", endpoint="https://a.com")
        await registry.update(tool.id, is_active=False)
        active = await registry.list_tools(active_only=True)
        all_tools = await registry.list_tools(active_only=False)
        assert len(active) == 0
        assert len(all_tools) == 1

    async def test_update_tool(self, registry: ToolRegistry):
        tool = await registry.register(name="update_me", endpoint="https://old.com")
        updated = await registry.update(tool.id, endpoint="https://new.com", description="Updated")
        assert updated is not None
        assert updated.endpoint == "https://new.com"
        assert updated.description == "Updated"

    async def test_update_nonexistent(self, registry: ToolRegistry):
        result = await registry.update("nonexistent-id", name="foo")
        assert result is None

    async def test_delete_tool(self, registry: ToolRegistry):
        tool = await registry.register(name="delete_me", endpoint="https://example.com")
        deleted = await registry.delete(tool.id)
        assert deleted is True
        assert await registry.get_by_id(tool.id) is None

    async def test_delete_nonexistent(self, registry: ToolRegistry):
        deleted = await registry.delete("nonexistent-id")
        assert deleted is False

    async def test_tool_count(self, registry: ToolRegistry):
        assert await registry.tool_count() == 0
        await registry.register(name="counted", endpoint="https://example.com")
        assert await registry.tool_count() == 1

    async def test_input_schema_roundtrip(self, registry: ToolRegistry):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        tool = await registry.register(name="schema_test", endpoint="https://a.com", input_schema=schema)
        found = await registry.get_by_name("schema_test")
        assert found is not None
        assert found.input_schema == schema


# ── Credential resolution ──────────────────────────────────────────────


class TestCredentialResolution:
    def test_env_var_resolution(self, monkeypatch):
        monkeypatch.setenv("MY_SECRET", "secret123")
        assert _resolve_credential("$MY_SECRET") == "secret123"

    def test_env_var_missing(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        assert _resolve_credential("$NONEXISTENT_VAR") == ""

    def test_literal_value(self):
        assert _resolve_credential("my-api-key-123") == "my-api-key-123"

    def test_empty_string(self):
        assert _resolve_credential("") == ""


# ── ToolExecutor tests ─────────────────────────────────────────────────


class TestToolExecutor:
    async def test_mock_mode(self, db: DatabaseManager, monkeypatch):
        monkeypatch.setenv("JANUS_MOCK_TOOLS", "true")
        registry = ToolRegistry(db)
        executor = ToolExecutor(registry=registry)
        assert executor.is_mock_mode is True

        result = await executor.execute("read_file", {"path": "/test.txt"})
        assert "content" in result  # mock response

    async def test_real_mode_unregistered_tool(self, db: DatabaseManager, monkeypatch):
        monkeypatch.delenv("JANUS_MOCK_TOOLS", raising=False)
        registry = ToolRegistry(db)
        executor = ToolExecutor(registry=registry)
        assert executor.is_mock_mode is False

        result = await executor.execute("nonexistent_tool", {})
        assert "error" in result
        assert "not registered" in result["error"]

    async def test_get_tool_definitions_mock(self, db: DatabaseManager, monkeypatch):
        monkeypatch.setenv("JANUS_MOCK_TOOLS", "true")
        registry = ToolRegistry(db)
        executor = ToolExecutor(registry=registry)
        defs = executor.get_tool_definitions()
        assert len(defs) > 0
        assert any(d["name"] == "read_file" for d in defs)

    async def test_get_tool_definitions_real(self, db: DatabaseManager, monkeypatch):
        monkeypatch.delenv("JANUS_MOCK_TOOLS", raising=False)
        registry = ToolRegistry(db)
        await registry.register(
            name="my_tool",
            description="A real tool",
            endpoint="https://example.com/tool",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        executor = ToolExecutor(registry=registry)
        defs = await executor.refresh_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "my_tool"

    async def test_tool_names_property(self, db: DatabaseManager, monkeypatch):
        monkeypatch.delenv("JANUS_MOCK_TOOLS", raising=False)
        registry = ToolRegistry(db)
        await registry.register(name="alpha", endpoint="https://a.com")
        await registry.register(name="beta", endpoint="https://b.com")
        executor = ToolExecutor(registry=registry)
        await executor.refresh_definitions()
        assert set(executor.tool_names) == {"alpha", "beta"}


# ── WebhookExecutor tests ─────────────────────────────────────────────


class TestWebhookExecutor:
    async def test_successful_call(self):
        tool = RegisteredTool(
            id="tool-test",
            name="test_webhook",
            endpoint="https://api.example.com/tool",
            auth_type="none",
        )
        executor = WebhookExecutor()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps({"result": "success"}).encode()

        with patch("janus.tools.executor.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await executor.call(tool, {"query": "test"})
            assert result == {"result": "success"}
            mock_client.request.assert_called_once()

    async def test_bearer_auth(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret-bearer-token")
        tool = RegisteredTool(
            id="tool-auth",
            name="auth_webhook",
            endpoint="https://api.example.com/tool",
            auth_type="bearer",
            auth_credential="$MY_TOKEN",
        )
        executor = WebhookExecutor()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"ok": true}'

        with patch("janus.tools.executor.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await executor.call(tool, {"data": "test"})

            call_kwargs = mock_client.request.call_args
            headers = call_kwargs.kwargs["headers"]
            assert headers["Authorization"] == "Bearer secret-bearer-token"

    async def test_api_key_auth(self):
        tool = RegisteredTool(
            id="tool-apikey",
            name="apikey_webhook",
            endpoint="https://api.example.com/tool",
            auth_type="api_key",
            auth_credential="raw-key-123",
        )
        executor = WebhookExecutor()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"ok": true}'

        with patch("janus.tools.executor.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await executor.call(tool, {})

            call_kwargs = mock_client.request.call_args
            headers = call_kwargs.kwargs["headers"]
            assert headers["X-API-Key"] == "raw-key-123"

    async def test_http_error_response(self):
        tool = RegisteredTool(
            id="tool-err",
            name="error_webhook",
            endpoint="https://api.example.com/tool",
        )
        executor = WebhookExecutor()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.content = b'{"detail": "server error"}'

        with patch("janus.tools.executor.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await executor.call(tool, {})
            assert "error" in result
            assert result["status_code"] == 500

    async def test_timeout(self):
        import httpx as httpx_mod

        tool = RegisteredTool(
            id="tool-timeout",
            name="slow_webhook",
            endpoint="https://api.example.com/slow",
            timeout_seconds=1.0,
        )
        executor = WebhookExecutor()

        with patch("janus.tools.executor.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=httpx_mod.TimeoutException("timed out"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await executor.call(tool, {})
            assert "error" in result
            assert "timed out" in result["error"]

    async def test_connect_error(self):
        import httpx as httpx_mod

        tool = RegisteredTool(
            id="tool-noconn",
            name="unreachable",
            endpoint="https://doesnotexist.invalid/tool",
        )
        executor = WebhookExecutor()

        with patch("janus.tools.executor.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=httpx_mod.ConnectError("connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await executor.call(tool, {})
            assert "error" in result
            assert "connect" in result["error"].lower()


# ── API Route tests ────────────────────────────────────────────────────


class TestToolRoutes:
    async def test_list_tools_empty(self, client: AsyncClient):
        resp = await client.get("/api/tools")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_register_webhook_tool(self, client: AsyncClient):
        resp = await client.post("/api/tools", json={
            "name": "my_search",
            "description": "Search my database",
            "type": "webhook",
            "endpoint": "https://api.myapp.com/tools/search",
            "auth_type": "bearer",
            "auth_credential": "$DB_TOKEN",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "my_search"
        assert data["type"] == "webhook"
        assert data["endpoint"] == "https://api.myapp.com/tools/search"
        assert data["id"].startswith("tool-")

    async def test_register_webhook_without_endpoint_fails(self, client: AsyncClient):
        resp = await client.post("/api/tools", json={
            "name": "no_endpoint",
            "type": "webhook",
        })
        assert resp.status_code == 400
        assert "endpoint" in resp.json()["detail"].lower()

    async def test_register_mcp_without_server_fails(self, client: AsyncClient):
        resp = await client.post("/api/tools", json={
            "name": "no_server",
            "type": "mcp",
        })
        assert resp.status_code == 400
        assert "server name" in resp.json()["detail"].lower()

    async def test_register_duplicate_fails(self, client: AsyncClient):
        await client.post("/api/tools", json={
            "name": "unique_tool",
            "type": "webhook",
            "endpoint": "https://a.com",
        })
        resp = await client.post("/api/tools", json={
            "name": "unique_tool",
            "type": "webhook",
            "endpoint": "https://b.com",
        })
        assert resp.status_code == 409

    async def test_get_tool_by_id(self, client: AsyncClient):
        create_resp = await client.post("/api/tools", json={
            "name": "get_me",
            "type": "webhook",
            "endpoint": "https://a.com",
        })
        tool_id = create_resp.json()["id"]

        resp = await client.get(f"/api/tools/{tool_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "get_me"

    async def test_get_tool_not_found(self, client: AsyncClient):
        resp = await client.get("/api/tools/nonexistent")
        assert resp.status_code == 404

    async def test_update_tool(self, client: AsyncClient):
        create_resp = await client.post("/api/tools", json={
            "name": "update_me",
            "type": "webhook",
            "endpoint": "https://old.com",
        })
        tool_id = create_resp.json()["id"]

        resp = await client.put(f"/api/tools/{tool_id}", json={
            "endpoint": "https://new.com",
            "description": "Updated description",
        })
        assert resp.status_code == 200
        assert resp.json()["endpoint"] == "https://new.com"
        assert resp.json()["description"] == "Updated description"

    async def test_delete_tool(self, client: AsyncClient):
        create_resp = await client.post("/api/tools", json={
            "name": "delete_me",
            "type": "webhook",
            "endpoint": "https://a.com",
        })
        tool_id = create_resp.json()["id"]

        resp = await client.delete(f"/api/tools/{tool_id}")
        assert resp.status_code == 204

        resp = await client.get(f"/api/tools/{tool_id}")
        assert resp.status_code == 404

    async def test_delete_not_found(self, client: AsyncClient):
        resp = await client.delete("/api/tools/nonexistent")
        assert resp.status_code == 404

    async def test_list_after_register(self, client: AsyncClient):
        await client.post("/api/tools", json={
            "name": "tool_1",
            "type": "webhook",
            "endpoint": "https://a.com",
        })
        await client.post("/api/tools", json={
            "name": "tool_2",
            "type": "webhook",
            "endpoint": "https://b.com",
        })
        resp = await client.get("/api/tools")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_test_tool_endpoint(self, client: AsyncClient):
        create_resp = await client.post("/api/tools", json={
            "name": "testable_tool",
            "type": "webhook",
            "endpoint": "https://api.example.com/run",
        })
        tool_id = create_resp.json()["id"]

        # Tool test will fail because endpoint doesn't exist — that's expected
        resp = await client.post(f"/api/tools/{tool_id}/test", json={
            "input": {"query": "hello"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["tool_name"] == "testable_tool"
        # Will have error since endpoint is fake
        assert data["success"] is False


# ── Mock mode fallback tests ──────────────────────────────────────────


class TestMockModeFallback:
    async def test_mock_mode_env_var(self, mock_client: AsyncClient):
        """When JANUS_MOCK_TOOLS=true, tool executor uses mock responses."""
        assert state.tool_executor is not None
        assert state.tool_executor.is_mock_mode is True

    async def test_mock_mode_returns_demo_tools(self, mock_client: AsyncClient):
        """In mock mode, get_tool_definitions returns DEMO_TOOLS."""
        defs = state.tool_executor.get_tool_definitions()
        tool_names = {d["name"] for d in defs}
        assert "read_file" in tool_names
        assert "execute_code" in tool_names
        assert "database_query" in tool_names

    async def test_mock_execute(self, mock_client: AsyncClient):
        """In mock mode, execute returns mock responses."""
        result = await state.tool_executor.execute("read_file", {"path": "/etc/passwd"})
        assert "content" in result
        assert "mock" in result["content"].lower() or "Contents" in result["content"]
