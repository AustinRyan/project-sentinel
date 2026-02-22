"""Real-time event broadcasting for WebSocket clients."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, AsyncIterator


@dataclass
class SecurityEvent:
    """A real-time security event pushed to WebSocket clients."""

    event_type: str
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "session_id": self.session_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class EventBroadcaster:
    """Pub/sub broadcaster that routes SecurityEvents to WebSocket subscribers by session."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[SecurityEvent]]] = defaultdict(list)

    async def publish(self, event: SecurityEvent) -> None:
        for queue in self._subscribers.get(event.session_id, []):
            await queue.put(event)

    async def subscribe(self, session_id: str) -> AsyncIterator[SecurityEvent]:
        queue: asyncio.Queue[SecurityEvent] = asyncio.Queue()
        self._subscribers[session_id].append(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self._subscribers[session_id].remove(queue)
            if not self._subscribers[session_id]:
                del self._subscribers[session_id]

    def subscriber_count(self, session_id: str) -> int:
        return len(self._subscribers.get(session_id, []))
