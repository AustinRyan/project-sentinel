"""Tests for the WebSocket event broadcaster."""
from __future__ import annotations

import asyncio

import pytest

from sentinel.web.events import EventBroadcaster, SecurityEvent


async def test_subscribe_receives_events() -> None:
    broadcaster = EventBroadcaster()
    received: list[SecurityEvent] = []

    async def listener():
        async for event in broadcaster.subscribe("session-1"):
            received.append(event)
            if len(received) >= 2:
                break

    task = asyncio.create_task(listener())
    await asyncio.sleep(0.05)

    event1 = SecurityEvent(
        event_type="verdict",
        session_id="session-1",
        data={"verdict": "allow", "risk_score": 5.0},
    )
    event2 = SecurityEvent(
        event_type="verdict",
        session_id="session-1",
        data={"verdict": "block", "risk_score": 85.0},
    )

    await broadcaster.publish(event1)
    await broadcaster.publish(event2)
    await asyncio.wait_for(task, timeout=2.0)

    assert len(received) == 2
    assert received[0].data["verdict"] == "allow"
    assert received[1].data["verdict"] == "block"


async def test_different_sessions_isolated() -> None:
    broadcaster = EventBroadcaster()
    received: list[SecurityEvent] = []

    async def listener():
        async for event in broadcaster.subscribe("session-A"):
            received.append(event)
            break

    task = asyncio.create_task(listener())
    await asyncio.sleep(0.05)

    await broadcaster.publish(SecurityEvent(
        event_type="verdict",
        session_id="session-B",
        data={"verdict": "allow"},
    ))

    await broadcaster.publish(SecurityEvent(
        event_type="verdict",
        session_id="session-A",
        data={"verdict": "block"},
    ))

    await asyncio.wait_for(task, timeout=2.0)
    assert len(received) == 1
    assert received[0].data["verdict"] == "block"


async def test_unsubscribe_cleans_up() -> None:
    broadcaster = EventBroadcaster()
    assert broadcaster.subscriber_count("session-1") == 0

    async def listener():
        async for _ in broadcaster.subscribe("session-1"):
            break

    task = asyncio.create_task(listener())
    await asyncio.sleep(0.05)
    assert broadcaster.subscriber_count("session-1") == 1

    await broadcaster.publish(SecurityEvent(
        event_type="verdict",
        session_id="session-1",
        data={},
    ))
    await asyncio.wait_for(task, timeout=2.0)
    await asyncio.sleep(0.05)
    assert broadcaster.subscriber_count("session-1") == 0


async def test_security_event_to_dict() -> None:
    event = SecurityEvent(
        event_type="verdict",
        session_id="s-1",
        data={"verdict": "allow"},
    )
    d = event.to_dict()
    assert d["event_type"] == "verdict"
    assert d["session_id"] == "s-1"
    assert "timestamp" in d
