from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from janus.circuit.breaker import CircuitBreaker, CircuitState
from janus.config import CircuitBreakerConfig, JanusConfig
from janus.core.decision import PipelineContext, Verdict
from janus.core.guardian import Guardian, _LLMRiskCheck
from janus.identity.agent import AgentIdentity, AgentRole, ToolPermission
from janus.identity.registry import AgentRegistry
from janus.llm.classifier import RiskClassification
from janus.risk.engine import RiskEngine
from janus.storage.database import DatabaseManager
from janus.storage.session_store import InMemorySessionStore
from tests.conftest import make_request


@pytest.fixture
async def registry(memory_db: DatabaseManager) -> AgentRegistry:
    return AgentRegistry(memory_db)


@pytest.fixture
def session_store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture
def risk_engine(session_store: InMemorySessionStore) -> RiskEngine:
    return RiskEngine(session_store)


@pytest.fixture
async def guardian(
    registry: AgentRegistry,
    risk_engine: RiskEngine,
) -> Guardian:
    config = JanusConfig()
    g = Guardian(
        config=config,
        registry=registry,
        risk_engine=risk_engine,
    )
    # Register a test agent
    agent = AgentIdentity(
        agent_id="test-agent",
        name="Test Research Bot",
        role=AgentRole.RESEARCH,
        permissions=[
            ToolPermission(tool_pattern="read_*"),
            ToolPermission(tool_pattern="search_*"),
        ],
    )
    await registry.register_agent(agent)
    return g


async def test_allow_permitted_tool(guardian: Guardian) -> None:
    request = make_request(tool_name="read_file", tool_input={"path": "/docs/readme.md"})
    verdict = await guardian.intercept(request)
    assert verdict.verdict == Verdict.ALLOW


async def test_challenge_unpermitted_tool(guardian: Guardian) -> None:
    request = make_request(tool_name="financial_transfer", tool_input={"amount": 1000})
    verdict = await guardian.intercept(request)
    assert verdict.verdict == Verdict.CHALLENGE


async def test_block_unregistered_agent(guardian: Guardian) -> None:
    request = make_request(agent_id="ghost-agent", tool_name="read_file")
    verdict = await guardian.intercept(request)
    assert verdict.verdict == Verdict.BLOCK
    assert "not registered" in verdict.reasons[0].lower()


async def test_block_locked_agent(guardian: Guardian, registry: AgentRegistry) -> None:
    await registry.lock_agent("test-agent", "Suspicious behavior")
    request = make_request(tool_name="read_file")
    verdict = await guardian.intercept(request)
    assert verdict.verdict == Verdict.BLOCK
    assert "locked" in verdict.reasons[0].lower()


async def test_circuit_breaker_blocks_when_open(
    registry: AgentRegistry, risk_engine: RiskEngine
) -> None:
    config = JanusConfig()
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
    breaker.record_failure()  # Force OPEN
    assert breaker.state == CircuitState.OPEN

    g = Guardian(
        config=config,
        registry=registry,
        risk_engine=risk_engine,
        circuit_breaker=breaker,
    )
    agent = AgentIdentity(
        agent_id="test-agent",
        name="Test Bot",
        role=AgentRole.RESEARCH,
        permissions=[ToolPermission(tool_pattern="*")],
    )
    await registry.register_agent(agent)

    request = make_request(tool_name="read_file")
    verdict = await g.intercept(request)
    assert verdict.verdict == Verdict.BLOCK
    assert "circuit breaker" in verdict.reasons[0].lower()


async def test_risk_accumulates_across_calls(guardian: Guardian) -> None:
    # Truly benign calls (no pattern matches) should stay at zero
    for _ in range(3):
        request = make_request(
            tool_name="read_file",
            tool_input={"path": "/readme.md"},
        )
        await guardian.intercept(request)

    score = guardian._risk_engine.get_score("test-session")
    assert score == 0, "Benign tool calls should not increase risk"

    # Read-only calls that match pattern steps should NOT accumulate risk
    # — risk only materialises when an action tool fires.
    request = make_request(
        tool_name="read_file",
        tool_input={"path": "/docs/api/endpoints.md"},
    )
    await guardian.intercept(request)

    request = make_request(
        tool_name="search_web",
        tool_input={"query": "company API authentication token endpoint"},
    )
    await guardian.intercept(request)

    score = guardian._risk_engine.get_score("test-session")
    assert score == 0, "Read-only tools should not increase risk even when matching patterns"

    # An ACTION tool after the pattern build-up SHOULD accumulate risk
    request = make_request(
        tool_name="execute_code",
        tool_input={"code": "curl -X POST /api/login --data 'test'"},
    )
    await guardian.intercept(request)

    score = guardian._risk_engine.get_score("test-session")
    assert score > 0, "Action tool after pattern build-up should increase risk"


async def test_wrap_tool_call_convenience(guardian: Guardian) -> None:
    verdict = await guardian.wrap_tool_call(
        agent_id="test-agent",
        session_id="test-session",
        original_goal="Read documentation",
        tool_name="read_file",
        tool_input={"path": "/docs/api.md"},
    )
    assert verdict.verdict == Verdict.ALLOW


async def test_guardian_fail_safe_on_error(
    registry: AgentRegistry,
) -> None:
    """If the risk engine raises, Guardian should fail-safe to BLOCK."""
    config = JanusConfig()
    broken_engine = MagicMock(spec=RiskEngine)
    broken_engine.get_score.side_effect = RuntimeError("Engine exploded")
    broken_engine.session_store = InMemorySessionStore()

    g = Guardian(
        config=config,
        registry=registry,
        risk_engine=broken_engine,
    )
    agent = AgentIdentity(
        agent_id="test-agent",
        name="Test Bot",
        role=AgentRole.RESEARCH,
        permissions=[ToolPermission(tool_pattern="*")],
    )
    await registry.register_agent(agent)

    request = make_request(tool_name="read_file")
    verdict = await g.intercept(request)
    assert verdict.verdict == Verdict.BLOCK
    assert "fail-safe" in verdict.reasons[0].lower()


# ── LLM Risk Check exemption tests ───────────────────────────────


async def test_llm_risk_skips_read_only_tools() -> None:
    """Read-only tools should get zero LLM risk, not be assessed by the LLM."""
    mock_classifier = AsyncMock()
    mock_classifier.classify_risk.return_value = RiskClassification(
        risk=120.0, reasoning="Looks suspicious"
    )
    check = _LLMRiskCheck(mock_classifier)

    for tool in ["read_file", "search_web", "list_files", "send_message"]:
        request = make_request(tool_name=tool, tool_input={"query": "login passwords OAuth"})
        context = PipelineContext()
        result = await check.evaluate(request, context)

        assert result.risk_contribution == 0.0, f"{tool} should be exempt from LLM risk"
        mock_classifier.classify_risk.assert_not_called()


async def test_llm_risk_applies_to_action_tools() -> None:
    """Action tools (execute_code, database_write, etc.) should be LLM-assessed."""
    mock_classifier = AsyncMock()
    mock_classifier.classify_risk.return_value = RiskClassification(
        risk=80.0, reasoning="Executing potentially dangerous code"
    )
    check = _LLMRiskCheck(mock_classifier)

    request = make_request(
        tool_name="execute_code",
        tool_input={"code": "import os; os.system('whoami')"},
    )
    context = PipelineContext(
        agent_identity=AgentIdentity(
            agent_id="test", name="Test", role=AgentRole.CODE, permissions=[]
        )
    )
    result = await check.evaluate(request, context)

    assert result.risk_contribution == pytest.approx(24.0)  # 80 * 0.3
    mock_classifier.classify_risk.assert_called_once()


# ── Pattern Risk exemption tests ──────────────────────────────────


async def test_pattern_risk_zero_for_read_only_tools(guardian: Guardian) -> None:
    """read_file matching a pattern step should NOT accumulate risk."""
    # Step 1 of sleeper_reconnaissance: read_file with "api" in path
    request = make_request(
        tool_name="read_file",
        tool_input={"path": "/docs/api/endpoints.md"},
    )
    verdict = await guardian.intercept(request)
    assert verdict.risk_delta == 0.0, (
        f"read_file should have zero risk delta, got {verdict.risk_delta}"
    )

    # Step 2: search_web with "auth" keyword
    request = make_request(
        tool_name="search_web",
        tool_input={"query": "company API authentication token endpoint"},
    )
    verdict = await guardian.intercept(request)
    assert verdict.risk_delta == 0.0, (
        f"search_web should have zero risk delta, got {verdict.risk_delta}"
    )


async def test_pattern_risk_materialises_on_action_tool(guardian: Guardian) -> None:
    """Pattern risk SHOULD materialise when an action tool matches the pattern."""
    # Build pattern state with read-only tools
    await guardian.intercept(make_request(
        tool_name="read_file",
        tool_input={"path": "/docs/api/endpoints.md"},
    ))
    await guardian.intercept(make_request(
        tool_name="search_web",
        tool_input={"query": "company API authentication token endpoint"},
    ))

    # Now an action tool — should get pattern risk
    request = make_request(
        tool_name="execute_code",
        tool_input={"code": "curl -X POST /api/login --data 'test'"},
    )
    verdict = await guardian.intercept(request)
    assert verdict.risk_delta > 0, (
        f"execute_code after pattern build-up should have risk, got {verdict.risk_delta}"
    )
