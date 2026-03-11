from __future__ import annotations

import os
from typing import Any

import pytest

# Ensure test signing key is set before any janus imports that touch licensing
os.environ.setdefault("JANUS_LICENSE_SECRET", "janus-test-key")

from janus.config import JanusConfig
from janus.core.decision import ToolCallRequest
from janus.licensing import generate_license
from janus.storage.database import DatabaseManager
from janus.storage.session_store import InMemorySessionStore

# Dynamically generated — uses the test signing key set above
TEST_PRO_KEY = generate_license(tier="pro", customer_id="test", expiry_days=36500)


@pytest.fixture
def config() -> JanusConfig:
    return JanusConfig()


@pytest.fixture
async def memory_db() -> DatabaseManager:
    """In-memory SQLite with schema applied."""
    db = DatabaseManager(":memory:")
    await db.connect()
    await db.apply_migrations()
    yield db  # type: ignore[misc]
    await db.close()


@pytest.fixture
def session_store() -> InMemorySessionStore:
    return InMemorySessionStore()


def make_request(**overrides: Any) -> ToolCallRequest:
    """Factory for ToolCallRequest with sensible defaults."""
    defaults: dict[str, Any] = {
        "agent_id": "test-agent",
        "session_id": "test-session",
        "tool_name": "read_file",
        "tool_input": {"path": "/tmp/test.txt"},
        "original_goal": "Read and summarize a document",
    }
    defaults.update(overrides)
    return ToolCallRequest(**defaults)
