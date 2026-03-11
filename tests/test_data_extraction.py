"""Tests for data extraction / volume tracking protections."""
from __future__ import annotations

from janus.config import JanusConfig
from janus.core.data_extraction import (
    DataVolumeCheck,
    DataVolumeTracker,
)
from janus.core.decision import PipelineContext, ToolCallRequest
from janus.core.guardian import Guardian
from janus.identity.agent import AgentIdentity, AgentRole, ToolPermission
from janus.identity.registry import AgentRegistry
from janus.risk.engine import RiskEngine
from janus.storage.database import DatabaseManager
from janus.storage.session_store import InMemorySessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    tool_name: str = "read_file",
    tool_input: dict | None = None,
    session_id: str = "test-session",
) -> ToolCallRequest:
    return ToolCallRequest(
        agent_id="test-agent",
        session_id=session_id,
        tool_name=tool_name,
        tool_input=tool_input or {"path": "/tmp/test.txt"},
    )


def _ctx() -> PipelineContext:
    return PipelineContext()


# ---------------------------------------------------------------------------
# DataVolumeTracker unit tests
# ---------------------------------------------------------------------------

def test_tracker_records_access():
    tracker = DataVolumeTracker()
    metrics = tracker.record_access("s1", "read_file", 1024)
    assert metrics.total_bytes_accessed == 1024
    assert metrics.read_count == 1


def test_tracker_accumulates_across_calls():
    tracker = DataVolumeTracker()
    for _ in range(5):
        tracker.record_access("s1", "read_file", 100 * 1024)
    metrics = tracker.get_metrics("s1")
    assert metrics.total_bytes_accessed == 500 * 1024
    assert metrics.read_count == 5


def test_tracker_sessions_isolated():
    tracker = DataVolumeTracker()
    tracker.record_access("session-a", "read_file", 100_000)
    tracker.record_access("session-b", "read_file", 200)
    assert tracker.get_metrics("session-a").total_bytes_accessed == 100_000
    assert tracker.get_metrics("session-b").total_bytes_accessed == 200


# ---------------------------------------------------------------------------
# DataVolumeCheck pipeline tests
# ---------------------------------------------------------------------------

async def test_normal_volume_passes():
    tracker = DataVolumeTracker()
    check = DataVolumeCheck(tracker)
    request = _make_request()
    result = await check.evaluate(request, _ctx())
    assert result.passed is True
    assert result.risk_contribution == 0.0


async def test_high_volume_flags_warning():
    tracker = DataVolumeTracker()
    check = DataVolumeCheck(tracker)
    # Pre-fill session with ~490 KB of reads
    for _ in range(49):
        tracker.record_access("warn-session", "read_file", 10 * 1024)

    # Action tool pushes past 500 KB — risk materialises on action tools
    request = _make_request(
        tool_name="api_call",
        tool_input={"url": "https://example.com/upload", "size": 20 * 1024},
        session_id="warn-session",
    )
    result = await check.evaluate(request, _ctx())
    assert result.risk_contribution > 0, "Should flag high volume on action tool"
    assert any("volume" in s.lower() for s in result.metadata.get("signals", []))


async def test_extreme_volume_blocks():
    tracker = DataVolumeTracker()
    check = DataVolumeCheck(tracker)
    # Pre-fill with ~1.9 MB of reads
    for _ in range(19):
        tracker.record_access("block-session", "read_file", 100 * 1024)

    # Action tool pushes past 2 MB — risk materialises on action tools
    request = _make_request(
        tool_name="send_email",
        tool_input={"to": "ext@example.com", "body": "data", "size": 200 * 1024},
        session_id="block-session",
    )
    result = await check.evaluate(request, _ctx())
    assert result.risk_contribution >= 30.0, f"Expected risk >= 30, got {result.risk_contribution}"


async def test_bulk_access_detection():
    tracker = DataVolumeTracker()
    check = DataVolumeCheck(tracker)
    # 11 rapid reads (small data, so volume check won't trigger)
    for i in range(11):
        tracker.record_access("bulk-session", "read_file", 100)

    # 12th call is an action tool — bulk access risk materialises
    request = _make_request(
        tool_name="api_call",
        tool_input={"url": "https://example.com/export"},
        session_id="bulk-session",
    )
    result = await check.evaluate(request, _ctx())
    assert result.risk_contribution > 0, "Should flag rapid sequential access on action tool"
    assert any("rapid" in s.lower() for s in result.metadata.get("signals", []))


async def test_large_query_without_limit():
    tracker = DataVolumeTracker()
    check = DataVolumeCheck(tracker)
    request = _make_request(
        tool_name="database_query",
        tool_input={"sql": "SELECT * FROM users"},
        session_id="query-session",
    )
    result = await check.evaluate(request, _ctx())
    assert result.risk_contribution > 0, "Unbounded SELECT * should flag risk"
    assert any("SELECT" in s or "LIMIT" in s or "Unbounded" in s for s in result.metadata.get("signals", []))


async def test_limited_query_passes():
    tracker = DataVolumeTracker()
    check = DataVolumeCheck(tracker)
    request = _make_request(
        tool_name="database_query",
        tool_input={"sql": "SELECT * FROM users LIMIT 10"},
        session_id="limited-query-session",
    )
    result = await check.evaluate(request, _ctx())
    # No unbounded query signal (volume is tiny)
    assert result.risk_contribution == 0.0


# ---------------------------------------------------------------------------
# Full Guardian pipeline integration
# ---------------------------------------------------------------------------

async def test_full_pipeline_bulk_reads(memory_db: DatabaseManager):
    """Bulk reads followed by an action tool should accumulate risk."""
    config = JanusConfig()
    registry = AgentRegistry(memory_db)
    session_store = InMemorySessionStore()
    risk_engine = RiskEngine(session_store)

    guardian = Guardian(
        config=config,
        registry=registry,
        risk_engine=risk_engine,
    )

    agent = AgentIdentity(
        agent_id="bulk-reader",
        name="Bulk Reader Bot",
        role=AgentRole.RESEARCH,
        permissions=[
            ToolPermission(tool_pattern="read_*"),
            ToolPermission(tool_pattern="api_call"),
        ],
    )
    await registry.register_agent(agent)

    sid = "bulk-pipeline-session"
    # Do 15 reads with substantial size hints (builds volume state)
    for i in range(15):
        await guardian.wrap_tool_call(
            agent_id="bulk-reader",
            session_id=sid,
            original_goal="Read all project files",
            tool_name="read_file",
            tool_input={"path": f"/project/file_{i}.py", "size": 40 * 1024},
        )

    # Read-only tools should not have accumulated risk
    score_before = risk_engine.get_score(sid)
    assert score_before == 0.0, f"Read-only calls should be 0, got {score_before}"

    # Action tool triggers risk from accumulated volume + bulk access
    await guardian.wrap_tool_call(
        agent_id="bulk-reader",
        session_id=sid,
        original_goal="Read all project files",
        tool_name="api_call",
        tool_input={"url": "https://example.com/export", "method": "POST"},
    )

    score = risk_engine.get_score(sid)
    assert score > 0, f"Action tool after 15 bulk reads should have risk, got {score}"
