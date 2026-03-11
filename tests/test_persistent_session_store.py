"""Tests for PersistentSessionStore — SQLite-backed session persistence."""
from __future__ import annotations

import asyncio

import pytest

from janus.storage.database import DatabaseManager
from janus.storage.persistent_session_store import PersistentSessionStore
from janus.storage.protocol import SessionStore
from janus.storage.session_store import InMemorySessionStore, RiskEvent


@pytest.fixture
async def db():
    db = DatabaseManager(":memory:")
    await db.connect()
    await db.apply_migrations()
    yield db
    await db.close()


@pytest.fixture
async def store(db):
    s = PersistentSessionStore(db, flush_interval=60.0)  # long interval; we flush manually
    await s.initialize()
    yield s
    await s.shutdown()


async def test_satisfies_protocol():
    """PersistentSessionStore satisfies the SessionStore protocol."""
    assert issubclass(PersistentSessionStore, SessionStore)


async def test_inmemory_satisfies_protocol():
    """InMemorySessionStore also satisfies the SessionStore protocol."""
    assert issubclass(InMemorySessionStore, SessionStore)


async def test_basic_session_operations(store: PersistentSessionStore):
    session = store.get_or_create_session("s1")
    assert session.session_id == "s1"
    assert store.get_risk_score("s1") == 0.0

    store.set_risk_score("s1", 42.5)
    assert store.get_risk_score("s1") == 42.5


async def test_goal_persistence(store: PersistentSessionStore):
    store.set_goal("s1", "Analyze quarterly data")
    assert store.get_goal("s1") == "Analyze quarterly data"
    # Second set_goal should be no-op (same as InMemorySessionStore behavior)
    store.set_goal("s1", "Different goal")
    assert store.get_goal("s1") == "Analyze quarterly data"


async def test_events_persist_immediately(store: PersistentSessionStore, db: DatabaseManager):
    event = RiskEvent(
        risk_delta=10.0,
        new_score=10.0,
        tool_name="execute_code",
        reason="dangerous payload",
    )
    store.add_event("s1", event)

    # Give the background task a moment to write
    await asyncio.sleep(0.1)

    rows = await db.fetchall(
        "SELECT * FROM session_events WHERE session_id = ?", ("s1",)
    )
    assert len(rows) == 1
    assert rows[0]["tool_name"] == "execute_code"
    assert rows[0]["risk_delta"] == 10.0


async def test_sessions_survive_restart(db: DatabaseManager):
    """Sessions persist across store instances sharing the same DB."""
    store1 = PersistentSessionStore(db, flush_interval=60.0)
    await store1.initialize()

    store1.get_or_create_session("s1")
    store1.set_risk_score("s1", 55.0)
    store1.set_goal("s1", "Research competitors")

    event = RiskEvent(
        risk_delta=55.0, new_score=55.0,
        tool_name="api_call", reason="test",
    )
    store1.add_event("s1", event)

    await store1.shutdown()  # flushes to DB

    # Create new store against same DB
    store2 = PersistentSessionStore(db, flush_interval=60.0)
    await store2.initialize()

    assert store2.get_risk_score("s1") == 55.0
    assert store2.get_goal("s1") == "Research competitors"
    assert len(store2.get_events("s1")) == 1

    await store2.shutdown()


async def test_flush_on_shutdown(store: PersistentSessionStore, db: DatabaseManager):
    store.get_or_create_session("s2")
    store.set_risk_score("s2", 30.0)
    store.set_goal("s2", "Test goal")

    await store.shutdown()

    row = await db.fetchone(
        "SELECT * FROM sessions WHERE session_id = ?", ("s2",)
    )
    assert row is not None
    assert row["risk_score"] == 30.0
    assert row["original_goal"] == "Test goal"


async def test_sessions_isolated(store: PersistentSessionStore):
    store.set_risk_score("s1", 10.0)
    store.set_risk_score("s2", 20.0)
    assert store.get_risk_score("s1") == 10.0
    assert store.get_risk_score("s2") == 20.0


async def test_list_sessions(store: PersistentSessionStore):
    store.get_or_create_session("a")
    store.get_or_create_session("b")
    store.get_or_create_session("c")
    sessions = store.list_sessions()
    assert set(sessions) == {"a", "b", "c"}


async def test_delete_session(store: PersistentSessionStore):
    store.get_or_create_session("tmp")
    store.set_risk_score("tmp", 50.0)
    store.delete_session("tmp")
    assert store.get_risk_score("tmp") == 0.0
    assert "tmp" not in store.list_sessions()


async def test_rapid_events(store: PersistentSessionStore, db: DatabaseManager):
    """1000 rapid events don't block."""
    for i in range(1000):
        event = RiskEvent(
            risk_delta=0.01,
            new_score=min(i * 0.01, 100.0),
            tool_name="read_file",
            reason=f"event-{i}",
        )
        store.add_event("rapid", event)

    # Events are in memory immediately
    assert len(store.get_events("rapid")) == 1000

    # Wait for async DB writes
    await asyncio.sleep(0.5)

    rows = await db.fetchall(
        "SELECT COUNT(*) as cnt FROM session_events WHERE session_id = ?",
        ("rapid",),
    )
    assert rows[0]["cnt"] == 1000


async def test_tool_call_history(store: PersistentSessionStore):
    store.record_tool_call("s1", "read_file", {"path": "/tmp/test"})
    store.record_tool_call("s1", "write_file", {"path": "/tmp/out"})
    history = store.get_tool_call_history("s1")
    assert len(history) == 2
    assert history[0] == ("read_file", {"path": "/tmp/test"})
