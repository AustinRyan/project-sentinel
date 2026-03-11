"""Tests for mock demo tools."""
from __future__ import annotations

from janus.web.tools import MockToolExecutor


async def test_all_tools_registered() -> None:
    executor = MockToolExecutor()
    assert "read_file" in executor.tool_names
    assert "search_web" in executor.tool_names
    assert "api_call" in executor.tool_names
    assert "execute_code" in executor.tool_names
    assert "write_file" in executor.tool_names
    assert "database_query" in executor.tool_names


async def test_read_file_returns_content() -> None:
    executor = MockToolExecutor()
    result = await executor.execute("read_file", {"path": "/docs/api.md"})
    assert "content" in result
    assert isinstance(result["content"], str)
    assert len(result["content"]) > 0


async def test_unknown_tool_returns_error() -> None:
    executor = MockToolExecutor()
    result = await executor.execute("nonexistent_tool", {})
    assert "error" in result


async def test_tool_definitions_for_claude() -> None:
    executor = MockToolExecutor()
    defs = executor.get_tool_definitions()
    assert isinstance(defs, list)
    assert len(defs) == len(executor.tool_names)
    for tool_def in defs:
        assert "name" in tool_def
        assert "description" in tool_def
        assert "input_schema" in tool_def


async def test_each_tool_returns_data() -> None:
    executor = MockToolExecutor()
    inputs = {
        "read_file": {"path": "/test.txt"},
        "search_web": {"query": "test"},
        "api_call": {"url": "https://example.com"},
        "execute_code": {"code": "print('hi')"},
        "write_file": {"path": "/tmp/f", "content": "hello"},
        "database_query": {"query": "SELECT 1"},
    }
    for tool_name, tool_input in inputs.items():
        result = await executor.execute(tool_name, tool_input)
        assert "error" not in result, f"{tool_name} returned error"
