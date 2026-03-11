"""Tests for upstream MCP connection manager."""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from mcp import types

from janus.mcp.client import UpstreamConnection, UpstreamManager
from janus.mcp.config import TransportType, UpstreamServerConfig


def _make_tools(names: list[str]) -> list[types.Tool]:
    return [
        types.Tool(name=n, description=f"Mock {n}", inputSchema={"type": "object"})
        for n in names
    ]


def _make_mock_session(tool_names: list[str]) -> AsyncMock:
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(
        return_value=MagicMock(tools=_make_tools(tool_names))
    )
    session.call_tool = AsyncMock(
        return_value=types.CallToolResult(
            content=[types.TextContent(type="text", text="mock result")]
        )
    )
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@asynccontextmanager
async def _mock_stdio_transport(*args, **kwargs):
    yield (AsyncMock(), AsyncMock())


# --- UpstreamConnection unit tests ---


def test_proxy_tool_name_no_namespace() -> None:
    conn = UpstreamConnection(
        config=UpstreamServerConfig(name="fs"),
        session=AsyncMock(),
        tools=_make_tools(["read_file"]),
    )
    assert conn.proxy_tool_name("read_file") == "read_file"


def test_proxy_tool_name_with_namespace() -> None:
    conn = UpstreamConnection(
        config=UpstreamServerConfig(name="gh", namespace="gh"),
        session=AsyncMock(),
        tools=_make_tools(["read_file"]),
    )
    assert conn.proxy_tool_name("read_file") == "gh__read_file"


def test_resolve_tool_name_no_namespace() -> None:
    conn = UpstreamConnection(
        config=UpstreamServerConfig(name="fs"),
        session=AsyncMock(),
        tools=_make_tools(["read_file", "write_file"]),
    )
    assert conn.resolve_tool_name("read_file") == "read_file"
    assert conn.resolve_tool_name("unknown") is None


def test_resolve_tool_name_with_namespace() -> None:
    conn = UpstreamConnection(
        config=UpstreamServerConfig(name="gh", namespace="gh"),
        session=AsyncMock(),
        tools=_make_tools(["create_issue"]),
    )
    assert conn.resolve_tool_name("gh__create_issue") == "create_issue"
    assert conn.resolve_tool_name("create_issue") is None
    assert conn.resolve_tool_name("other__create_issue") is None


# --- UpstreamManager tests ---


async def test_connect_discovers_tools() -> None:
    session = _make_mock_session(["read_file", "write_file"])

    with (
        patch("janus.mcp.client.stdio_client", _mock_stdio_transport),
        patch("janus.mcp.client.ClientSession", return_value=session),
    ):
        manager = UpstreamManager()
        await manager.connect([
            UpstreamServerConfig(name="fs", command="echo", transport=TransportType.STDIO),
        ])

        tools = manager.get_all_tools()
        assert len(tools) == 2
        assert tools[0].name == "read_file"
        assert tools[1].name == "write_file"
        assert manager.tool_count == 2
        assert "fs" in manager.server_names

        await manager.close()


async def test_namespace_prefixing() -> None:
    session = _make_mock_session(["create_issue", "list_repos"])

    with (
        patch("janus.mcp.client.stdio_client", _mock_stdio_transport),
        patch("janus.mcp.client.ClientSession", return_value=session),
    ):
        manager = UpstreamManager()
        await manager.connect([
            UpstreamServerConfig(
                name="github", command="echo",
                transport=TransportType.STDIO, namespace="gh",
            ),
        ])

        tools = manager.get_all_tools()
        assert tools[0].name == "gh__create_issue"
        assert tools[1].name == "gh__list_repos"
        assert "[github]" in tools[0].description

        await manager.close()


async def test_multiple_upstreams() -> None:
    session_fs = _make_mock_session(["read_file"])
    session_gh = _make_mock_session(["create_issue"])

    call_count = 0

    def session_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return session_fs if call_count == 1 else session_gh

    with (
        patch("janus.mcp.client.stdio_client", _mock_stdio_transport),
        patch("janus.mcp.client.ClientSession", side_effect=session_factory),
    ):
        manager = UpstreamManager()
        await manager.connect([
            UpstreamServerConfig(name="fs", command="echo"),
            UpstreamServerConfig(name="gh", command="echo", namespace="gh"),
        ])

        tools = manager.get_all_tools()
        names = [t.name for t in tools]
        assert "read_file" in names
        assert "gh__create_issue" in names
        assert manager.tool_count == 2

        await manager.close()


async def test_call_tool_forwards() -> None:
    session = _make_mock_session(["read_file"])

    with (
        patch("janus.mcp.client.stdio_client", _mock_stdio_transport),
        patch("janus.mcp.client.ClientSession", return_value=session),
    ):
        manager = UpstreamManager()
        await manager.connect([
            UpstreamServerConfig(name="fs", command="echo"),
        ])

        result = await manager.call_tool("read_file", {"path": "/test.txt"})
        assert not result.isError
        assert result.content[0].text == "mock result"
        session.call_tool.assert_called_once_with("read_file", {"path": "/test.txt"})

        await manager.close()


async def test_call_unknown_tool() -> None:
    manager = UpstreamManager()
    result = await manager.call_tool("nonexistent", {})
    assert result.isError is True
    assert "Unknown tool" in result.content[0].text


async def test_connect_failure_graceful() -> None:
    @asynccontextmanager
    async def _failing_transport(*args, **kwargs):
        raise ConnectionError("boom")
        yield  # pragma: no cover

    with (
        patch("janus.mcp.client.stdio_client", _failing_transport),
    ):
        manager = UpstreamManager()
        await manager.connect([
            UpstreamServerConfig(name="bad", command="nope"),
        ])

        # Should not crash, just skip the bad upstream
        assert manager.tool_count == 0
        assert manager.server_names == []

        await manager.close()


async def test_close_clears_state() -> None:
    session = _make_mock_session(["read_file"])

    with (
        patch("janus.mcp.client.stdio_client", _mock_stdio_transport),
        patch("janus.mcp.client.ClientSession", return_value=session),
    ):
        manager = UpstreamManager()
        await manager.connect([
            UpstreamServerConfig(name="fs", command="echo"),
        ])
        assert manager.tool_count == 1

        await manager.close()
        assert manager.tool_count == 0
        assert manager.server_names == []
