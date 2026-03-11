from __future__ import annotations

from datetime import UTC, datetime, timedelta

import time_machine

from janus.identity.agent import AgentIdentity, AgentRole
from janus.identity.registry import AgentRegistry
from janus.itdr.anomaly import ServiceAccountAnomalyDetector
from janus.itdr.collusion import CrossAgentCollusionDetector
from janus.itdr.escalation import PrivilegeEscalationTracker
from janus.itdr.signals import AnomalySignal, CollusionSignal, EscalationSignal
from janus.storage.database import DatabaseManager
from janus.storage.models import ToolUsageRow
from tests.conftest import make_request

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    agent_id: str = "agent-1",
    name: str = "Test Agent",
    role: AgentRole = AgentRole.RESEARCH,
) -> AgentIdentity:
    return AgentIdentity(agent_id=agent_id, name=name, role=role)


def _make_usage_row(
    agent_id: str = "agent-1",
    tool_name: str = "read_file",
    session_id: str = "test-session",
    timestamp: str = "2024-01-01T10:00:00+00:00",
) -> ToolUsageRow:
    return ToolUsageRow(
        agent_id=agent_id,
        tool_name=tool_name,
        session_id=session_id,
        timestamp=timestamp,
    )


# ===========================================================================
# Anomaly detection tests
# ===========================================================================


async def test_anomaly_unusual_hour(memory_db: DatabaseManager) -> None:
    """A request at 3am UTC should trigger an unusual_hour anomaly."""
    registry = AgentRegistry(memory_db)
    detector = ServiceAccountAnomalyDetector(registry)
    agent = _make_agent()

    ts = datetime(2024, 1, 1, 3, 0, tzinfo=UTC)
    request = make_request(
        agent_id="agent-1",
        tool_name="read_file",
        timestamp=ts,
    )
    # Provide read_file in history so only the hour anomaly fires
    history = [_make_usage_row(tool_name="read_file")]

    signal = await detector.check(request, agent, history)
    assert signal is not None
    assert isinstance(signal, AnomalySignal)
    assert "unusual_hour" in signal.anomaly_types


async def test_anomaly_normal_hour(memory_db: DatabaseManager) -> None:
    """A request at 10am UTC with known tool and low volume should be clean."""
    registry = AgentRegistry(memory_db)
    detector = ServiceAccountAnomalyDetector(registry)
    agent = _make_agent()

    ts = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    request = make_request(
        agent_id="agent-1",
        tool_name="read_file",
        timestamp=ts,
    )
    history = [_make_usage_row(tool_name="read_file")]

    signal = await detector.check(request, agent, history)
    assert signal is None


async def test_anomaly_new_endpoint(memory_db: DatabaseManager) -> None:
    """A tool that has never been used before should trigger new_endpoint."""
    registry = AgentRegistry(memory_db)
    detector = ServiceAccountAnomalyDetector(registry)
    agent = _make_agent()

    ts = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    request = make_request(
        agent_id="agent-1",
        tool_name="delete_database",
        timestamp=ts,
    )
    # History only has read_file
    history = [_make_usage_row(tool_name="read_file")]

    signal = await detector.check(request, agent, history)
    assert signal is not None
    assert "new_endpoint" in signal.anomaly_types


async def test_anomaly_known_endpoint(memory_db: DatabaseManager) -> None:
    """A tool that has been used before should NOT trigger new_endpoint."""
    registry = AgentRegistry(memory_db)
    detector = ServiceAccountAnomalyDetector(registry)
    agent = _make_agent()

    ts = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    request = make_request(
        agent_id="agent-1",
        tool_name="read_file",
        timestamp=ts,
    )
    history = [_make_usage_row(tool_name="read_file")]

    signal = await detector.check(request, agent, history)
    assert signal is None


async def test_anomaly_volume_spike(memory_db: DatabaseManager) -> None:
    """More than 10 calls in the same session should trigger volume_spike."""
    registry = AgentRegistry(memory_db)
    detector = ServiceAccountAnomalyDetector(registry)
    agent = _make_agent()

    ts = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    request = make_request(
        agent_id="agent-1",
        tool_name="read_file",
        session_id="busy-session",
        timestamp=ts,
    )
    # 11 previous calls in the same session (exceeds threshold of 10)
    history = [
        _make_usage_row(tool_name="read_file", session_id="busy-session")
        for _ in range(11)
    ]

    signal = await detector.check(request, agent, history)
    assert signal is not None
    assert "volume_spike" in signal.anomaly_types


async def test_anomaly_severity_scaling(memory_db: DatabaseManager) -> None:
    """More anomaly types should produce a higher severity."""
    registry = AgentRegistry(memory_db)
    detector = ServiceAccountAnomalyDetector(registry)
    agent = _make_agent()

    # Trigger all 3: unusual hour (3am), new endpoint, volume spike
    ts = datetime(2024, 1, 1, 3, 0, tzinfo=UTC)
    request = make_request(
        agent_id="agent-1",
        tool_name="never_seen_tool",
        session_id="busy-session",
        timestamp=ts,
    )
    history = [
        _make_usage_row(tool_name="read_file", session_id="busy-session")
        for _ in range(11)
    ]

    signal = await detector.check(request, agent, history)
    assert signal is not None
    assert len(signal.anomaly_types) == 3
    assert signal.severity == "high"

    # Trigger 2: unusual hour + new endpoint (low volume, different session)
    request_2 = make_request(
        agent_id="agent-1",
        tool_name="never_seen_tool",
        session_id="quiet-session",
        timestamp=ts,
    )
    history_2 = [_make_usage_row(tool_name="read_file", session_id="quiet-session")]
    signal_2 = await detector.check(request_2, agent, history_2)
    assert signal_2 is not None
    assert len(signal_2.anomaly_types) == 2
    assert signal_2.severity == "medium"

    # Trigger 1: just unusual hour
    request_3 = make_request(
        agent_id="agent-1",
        tool_name="read_file",
        session_id="quiet-session",
        timestamp=ts,
    )
    signal_3 = await detector.check(request_3, agent, history_2)
    assert signal_3 is not None
    assert len(signal_3.anomaly_types) == 1
    assert signal_3.severity == "low"


# ===========================================================================
# Collusion detection tests
# ===========================================================================


def test_collusion_detected() -> None:
    """Agent B referencing data fingerprint previously read by Agent A."""
    detector = CrossAgentCollusionDetector()

    # Agent A reads some data
    detector.record_data_access(
        agent_id="agent-a",
        data_fingerprint="fp-secret-123",
        access_type="read",
        session_id="session-a",
    )

    # Agent B sends a request whose tool_input contains the same fingerprint
    request = make_request(
        agent_id="agent-b",
        tool_name="send_email",
        tool_input={"body": "fp-secret-123", "to": "external@example.com"},
    )

    signal = detector.check(request)
    assert signal is not None
    assert isinstance(signal, CollusionSignal)
    assert signal.severity == "high"
    assert signal.source_agent_id == "agent-a"
    assert signal.target_agent_id == "agent-b"
    assert signal.data_fingerprint == "fp-secret-123"


def test_collusion_not_detected_same_agent() -> None:
    """Same agent reading its own data should NOT trigger collusion."""
    detector = CrossAgentCollusionDetector()

    detector.record_data_access(
        agent_id="agent-a",
        data_fingerprint="fp-my-data",
        access_type="read",
        session_id="session-a",
    )

    # Same agent references the data
    request = make_request(
        agent_id="agent-a",
        tool_name="process_data",
        tool_input={"data": "fp-my-data"},
    )

    signal = detector.check(request)
    assert signal is None


def test_collusion_no_match() -> None:
    """No matching fingerprints in tool_input means no collusion signal."""
    detector = CrossAgentCollusionDetector()

    detector.record_data_access(
        agent_id="agent-a",
        data_fingerprint="fp-secret-123",
        access_type="read",
        session_id="session-a",
    )

    # Agent B uses completely different data
    request = make_request(
        agent_id="agent-b",
        tool_name="read_file",
        tool_input={"path": "/tmp/unrelated.txt"},
    )

    signal = detector.check(request)
    assert signal is None


# ===========================================================================
# Privilege escalation tracking tests
# ===========================================================================


@time_machine.travel(datetime(2024, 6, 15, 12, 0, tzinfo=UTC), tick=False)
def test_escalation_no_attempts() -> None:
    """No recorded attempts should return None."""
    tracker = PrivilegeEscalationTracker()

    signal = tracker.check("agent-1")
    assert signal is None


@time_machine.travel(datetime(2024, 6, 15, 12, 0, tzinfo=UTC), tick=False)
def test_escalation_few_attempts() -> None:
    """1-2 attempts should produce a low-severity signal."""
    tracker = PrivilegeEscalationTracker()

    tracker.record_attempt("agent-1", "delete_database")
    tracker.record_attempt("agent-1", "admin_panel")

    signal = tracker.check("agent-1")
    assert signal is not None
    assert isinstance(signal, EscalationSignal)
    assert signal.severity == "low"
    assert len(signal.attempts) == 2


@time_machine.travel(datetime(2024, 6, 15, 12, 0, tzinfo=UTC), tick=False)
def test_escalation_many_attempts() -> None:
    """3+ attempts should produce a high-severity signal."""
    tracker = PrivilegeEscalationTracker()

    tracker.record_attempt("agent-1", "delete_database")
    tracker.record_attempt("agent-1", "admin_panel")
    tracker.record_attempt("agent-1", "modify_permissions")

    signal = tracker.check("agent-1")
    assert signal is not None
    assert isinstance(signal, EscalationSignal)
    assert signal.severity == "high"
    assert len(signal.attempts) == 3


@time_machine.travel(datetime(2024, 6, 15, 12, 0, tzinfo=UTC), tick=False)
def test_escalation_window_expiry() -> None:
    """Attempts older than the window should not be counted."""
    tracker = PrivilegeEscalationTracker()

    # Record 3 attempts, but manually set their timestamps to 2 hours ago
    tracker.record_attempt("agent-1", "delete_database")
    tracker.record_attempt("agent-1", "admin_panel")
    tracker.record_attempt("agent-1", "modify_permissions")

    # Manually backdate all attempts to 2 hours ago
    old_time = datetime.now(UTC) - timedelta(hours=2)
    for attempt in tracker._boundary_attempts["agent-1"]:
        attempt.timestamp = old_time

    # With a 30-minute window, these old attempts should not count
    signal = tracker.check("agent-1", window_minutes=30.0)
    assert signal is None
