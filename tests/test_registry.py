from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from janus.core.exceptions import AgentAlreadyExistsError, AgentNotFoundError
from janus.identity.agent import AgentIdentity, AgentRole, ToolPermission
from janus.identity.challenge import IdentityChallenger
from janus.identity.credential import CredentialManager
from janus.identity.registry import AgentRegistry
from janus.storage.database import DatabaseManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    agent_id: str = "agent-1",
    name: str = "Test Agent",
    role: AgentRole = AgentRole.RESEARCH,
    permissions: list[ToolPermission] | None = None,
    credential_hash: str = "",
    credential_expires_at: datetime | None = None,
    credential_last_rotated: datetime | None = None,
    is_locked: bool = False,
    lock_reason: str = "",
    metadata: dict[str, str] | None = None,
) -> AgentIdentity:
    return AgentIdentity(
        agent_id=agent_id,
        name=name,
        role=role,
        permissions=permissions or [],
        created_at=datetime.now(UTC),
        credential_hash=credential_hash,
        credential_expires_at=credential_expires_at,
        credential_last_rotated=credential_last_rotated,
        is_locked=is_locked,
        lock_reason=lock_reason,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Registration & retrieval
# ---------------------------------------------------------------------------


async def test_register_and_retrieve_agent(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    agent = _make_agent(
        permissions=[ToolPermission(tool_pattern="read_*")],
        metadata={"team": "alpha"},
    )
    await registry.register_agent(agent)

    retrieved = await registry.get_agent("agent-1")
    assert retrieved is not None
    assert retrieved.agent_id == "agent-1"
    assert retrieved.name == "Test Agent"
    assert retrieved.role == AgentRole.RESEARCH
    assert len(retrieved.permissions) == 1
    assert retrieved.permissions[0].tool_pattern == "read_*"
    assert retrieved.permissions[0].allowed is True
    assert retrieved.permissions[0].requires_sandbox is False
    assert retrieved.metadata == {"team": "alpha"}
    assert retrieved.is_locked is False


async def test_register_duplicate_raises(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    agent = _make_agent()
    await registry.register_agent(agent)

    with pytest.raises(AgentAlreadyExistsError):
        await registry.register_agent(agent)


async def test_get_agent_not_found(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    result = await registry.get_agent("nonexistent")
    assert result is None


# ---------------------------------------------------------------------------
# Lock / unlock
# ---------------------------------------------------------------------------


async def test_lock_unlock_agent(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    agent = _make_agent()
    await registry.register_agent(agent)

    await registry.lock_agent("agent-1", "suspicious activity")
    locked = await registry.get_agent("agent-1")
    assert locked is not None
    assert locked.is_locked is True
    assert locked.lock_reason == "suspicious activity"

    await registry.unlock_agent("agent-1")
    unlocked = await registry.get_agent("agent-1")
    assert unlocked is not None
    assert unlocked.is_locked is False
    assert unlocked.lock_reason == ""


async def test_lock_nonexistent_agent_raises(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    with pytest.raises(AgentNotFoundError):
        await registry.lock_agent("ghost", "reason")


# ---------------------------------------------------------------------------
# Permission glob matching
# ---------------------------------------------------------------------------


async def test_permission_glob_matching(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    agent = _make_agent(permissions=[ToolPermission(tool_pattern="read_*")])

    assert registry.check_permission(agent, "read_file") is True
    assert registry.check_permission(agent, "read_database") is True
    assert registry.check_permission(agent, "write_file") is False


async def test_permission_wildcard_matches_all(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    agent = _make_agent(permissions=[ToolPermission(tool_pattern="*")])

    assert registry.check_permission(agent, "read_file") is True
    assert registry.check_permission(agent, "write_file") is True
    assert registry.check_permission(agent, "delete_everything") is True


async def test_permission_denied_pattern(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    agent = _make_agent(
        permissions=[ToolPermission(tool_pattern="delete_*", allowed=False)]
    )
    # Denied pattern should not count as a positive match
    assert registry.check_permission(agent, "delete_file") is False


async def test_permission_no_patterns(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    agent = _make_agent(permissions=[])
    assert registry.check_permission(agent, "anything") is False


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


async def test_list_agents_by_role(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)

    await registry.register_agent(_make_agent(agent_id="r1", role=AgentRole.RESEARCH))
    await registry.register_agent(_make_agent(agent_id="r2", role=AgentRole.RESEARCH))
    await registry.register_agent(_make_agent(agent_id="c1", role=AgentRole.CODE))

    research_agents = await registry.list_agents(role=AgentRole.RESEARCH)
    assert len(research_agents) == 2
    assert {a.agent_id for a in research_agents} == {"r1", "r2"}

    code_agents = await registry.list_agents(role=AgentRole.CODE)
    assert len(code_agents) == 1
    assert code_agents[0].agent_id == "c1"

    all_agents = await registry.list_agents()
    assert len(all_agents) == 3


# ---------------------------------------------------------------------------
# Tool-usage tracking
# ---------------------------------------------------------------------------


async def test_record_and_get_tool_usage(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    agent = _make_agent()
    await registry.register_agent(agent)

    await registry.record_tool_usage("agent-1", "read_file", "sess-1", 15.0)
    await registry.record_tool_usage("agent-1", "write_file", "sess-1", 25.0)

    usage = await registry.get_tool_usage("agent-1")
    assert len(usage) == 2
    assert usage[0].tool_name == "read_file"
    assert usage[0].risk_score_at_time == 15.0
    assert usage[1].tool_name == "write_file"


async def test_get_tool_usage_since(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    agent = _make_agent()
    await registry.register_agent(agent)

    await registry.record_tool_usage("agent-1", "read_file", "sess-1", 10.0)

    # Query with a far-future cutoff should return nothing
    future = datetime.now(UTC) + timedelta(hours=1)
    usage = await registry.get_tool_usage("agent-1", since=future)
    assert len(usage) == 0

    # Query with a past cutoff should return everything
    past = datetime.now(UTC) - timedelta(hours=1)
    usage = await registry.get_tool_usage("agent-1", since=past)
    assert len(usage) == 1


# ---------------------------------------------------------------------------
# Credential rotation & expiry
# ---------------------------------------------------------------------------


async def test_credential_rotation(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    agent = _make_agent()
    await registry.register_agent(agent)

    cred_mgr = CredentialManager(registry)
    new_hash = await cred_mgr.rotate_credential("agent-1", "super-secret-key")

    expected_hash = hashlib.sha256(b"super-secret-key").hexdigest()
    assert new_hash == expected_hash

    updated = await registry.get_agent("agent-1")
    assert updated is not None
    assert updated.credential_hash == expected_hash
    assert updated.credential_expires_at is not None
    assert updated.credential_last_rotated is not None

    # Should count as recently rotated
    assert cred_mgr.was_recently_rotated(updated) is True


async def test_credential_expiry_check(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    cred_mgr = CredentialManager(registry)

    # Agent with no expiry -> not expired
    agent_no_expiry = _make_agent(agent_id="no-exp")
    assert cred_mgr.is_expired(agent_no_expiry) is False

    # Agent with future expiry -> not expired
    agent_future = _make_agent(
        agent_id="future-exp",
        credential_expires_at=datetime.now(UTC) + timedelta(days=30),
    )
    assert cred_mgr.is_expired(agent_future) is False

    # Agent with past expiry -> expired
    agent_past = _make_agent(
        agent_id="past-exp",
        credential_expires_at=datetime.now(UTC) - timedelta(days=1),
    )
    assert cred_mgr.is_expired(agent_past) is True


async def test_credential_not_recently_rotated(memory_db: DatabaseManager) -> None:
    registry = AgentRegistry(memory_db)
    cred_mgr = CredentialManager(registry)

    # No rotation timestamp -> not recently rotated
    agent = _make_agent()
    assert cred_mgr.was_recently_rotated(agent) is False

    # Rotated long ago -> not recently rotated
    old_agent = _make_agent(
        agent_id="old",
        credential_last_rotated=datetime.now(UTC) - timedelta(hours=48),
    )
    assert cred_mgr.was_recently_rotated(old_agent) is False


# ---------------------------------------------------------------------------
# Identity challenge
# ---------------------------------------------------------------------------


async def test_challenge_passes_with_matching_permission(
    memory_db: DatabaseManager,
) -> None:
    challenger = IdentityChallenger()
    agent = _make_agent(permissions=[ToolPermission(tool_pattern="read_*")])

    result = challenger.challenge(agent, "read_file")
    assert result.passed is True
    assert result.confidence == 1.0
    assert "read_*" in result.reasoning


async def test_challenge_fails_with_no_matching_permission(
    memory_db: DatabaseManager,
) -> None:
    challenger = IdentityChallenger()
    agent = _make_agent(permissions=[ToolPermission(tool_pattern="read_*")])

    result = challenger.challenge(agent, "write_file")
    assert result.passed is False
    assert "does not match" in result.reasoning


async def test_challenge_fails_with_denied_pattern(
    memory_db: DatabaseManager,
) -> None:
    challenger = IdentityChallenger()
    agent = _make_agent(
        permissions=[ToolPermission(tool_pattern="delete_*", allowed=False)]
    )

    result = challenger.challenge(agent, "delete_file")
    assert result.passed is False
    assert result.confidence == 1.0
    assert "denied" in result.reasoning
