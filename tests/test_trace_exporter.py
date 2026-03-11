"""Tests for audit log export — TraceExporter CSV/JSON/JSONL."""
from __future__ import annotations

import csv
import io
import json

import pytest

from janus.forensics.exporter import TraceExporter
from janus.storage.database import DatabaseManager


@pytest.fixture
async def db():
    db = DatabaseManager(":memory:")
    await db.connect()
    await db.apply_migrations()
    yield db
    await db.close()


async def _insert_trace(db: DatabaseManager, **overrides) -> None:
    defaults = {
        "trace_id": "t-1",
        "session_id": "s-1",
        "agent_id": "agent-1",
        "request_id": "r-1",
        "tool_name": "execute_code",
        "tool_input_json": '{"code": "print(1)"}',
        "verdict": "block",
        "risk_score": 85.0,
        "risk_delta": 20.0,
        "drift_score": 0.3,
        "reasons_json": '["dangerous payload"]',
        "itdr_signals_json": "[]",
        "explanation": "Blocked due to dangerous code execution",
        "original_goal": "Research task",
        "conversation_context_json": "[]",
        "timestamp": "2026-01-15T10:00:00",
    }
    defaults.update(overrides)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join("?" for _ in defaults)
    await db.execute(
        f"INSERT INTO security_traces ({cols}) VALUES ({placeholders})",
        tuple(defaults.values()),
    )
    await db.commit()


async def test_query_all_traces(db: DatabaseManager):
    await _insert_trace(db, trace_id="t-1")
    await _insert_trace(db, trace_id="t-2", verdict="allow", risk_score=5.0)

    exporter = TraceExporter(db)
    traces = await exporter.query_traces()
    assert len(traces) == 2


async def test_filter_by_verdict(db: DatabaseManager):
    await _insert_trace(db, trace_id="t-1", verdict="block")
    await _insert_trace(db, trace_id="t-2", verdict="allow")

    exporter = TraceExporter(db)
    traces = await exporter.query_traces(verdict="block")
    assert len(traces) == 1
    assert traces[0]["verdict"] == "block"


async def test_filter_by_date_range(db: DatabaseManager):
    await _insert_trace(db, trace_id="t-1", timestamp="2026-01-10T10:00:00")
    await _insert_trace(db, trace_id="t-2", timestamp="2026-02-15T10:00:00")

    exporter = TraceExporter(db)
    traces = await exporter.query_traces(date_from="2026-02-01")
    assert len(traces) == 1
    assert traces[0]["trace_id"] == "t-2"


async def test_filter_by_agent(db: DatabaseManager):
    await _insert_trace(db, trace_id="t-1", agent_id="a1")
    await _insert_trace(db, trace_id="t-2", agent_id="a2")

    exporter = TraceExporter(db)
    traces = await exporter.query_traces(agent_id="a1")
    assert len(traces) == 1


async def test_filter_by_session(db: DatabaseManager):
    await _insert_trace(db, trace_id="t-1", session_id="s1")
    await _insert_trace(db, trace_id="t-2", session_id="s2")

    exporter = TraceExporter(db)
    traces = await exporter.query_traces(session_id="s1")
    assert len(traces) == 1


async def test_filter_by_min_risk(db: DatabaseManager):
    await _insert_trace(db, trace_id="t-1", risk_score=10.0)
    await _insert_trace(db, trace_id="t-2", risk_score=90.0)

    exporter = TraceExporter(db)
    traces = await exporter.query_traces(min_risk=50.0)
    assert len(traces) == 1
    assert traces[0]["risk_score"] == 90.0


async def test_limit_enforcement(db: DatabaseManager):
    for i in range(10):
        await _insert_trace(db, trace_id=f"t-{i}")

    exporter = TraceExporter(db)
    traces = await exporter.query_traces(limit=3)
    assert len(traces) == 3


async def test_empty_results(db: DatabaseManager):
    exporter = TraceExporter(db)
    traces = await exporter.query_traces()
    assert traces == []


async def test_to_json_format(db: DatabaseManager):
    await _insert_trace(db)
    exporter = TraceExporter(db)
    traces = await exporter.query_traces()

    result = exporter.to_json(traces)
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["verdict"] == "block"


async def test_to_jsonl_format(db: DatabaseManager):
    await _insert_trace(db, trace_id="t-1")
    await _insert_trace(db, trace_id="t-2")

    exporter = TraceExporter(db)
    traces = await exporter.query_traces()

    result = exporter.to_jsonl(traces)
    lines = result.strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert "trace_id" in parsed


async def test_to_csv_format(db: DatabaseManager):
    await _insert_trace(db)
    exporter = TraceExporter(db)
    traces = await exporter.query_traces()

    result = exporter.to_csv(traces)
    reader = csv.DictReader(io.StringIO(result))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "block"
    assert "trace_id" in rows[0]


async def test_csv_headers_match_fields(db: DatabaseManager):
    await _insert_trace(db)
    exporter = TraceExporter(db)
    traces = await exporter.query_traces()

    result = exporter.to_csv(traces)
    header_line = result.split("\n")[0]
    headers = header_line.strip().split(",")
    assert "trace_id" in headers
    assert "verdict" in headers
    assert "risk_score" in headers
    assert "timestamp" in headers


async def test_csv_empty_returns_empty():
    result = TraceExporter.to_csv([])
    assert result == ""
