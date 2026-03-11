"""Testbed backend: 4-agent manual testing environment.

FastAPI app on port 8001 with FREE and PRO tier agents,
SDK and MCP integration paths. Mirrors janus/web/app.py API shape.
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from janus.config import JanusConfig
from janus.core.guardian import Guardian
from janus.drift.detector import SemanticDriftDetector
from janus.identity.agent import AgentIdentity, AgentRole, ToolPermission
from janus.identity.registry import AgentRegistry
from janus.licensing import generate_license
from janus.llm.classifier import SecurityClassifier
from janus.llm.client import AnthropicClientWrapper
from janus.mcp.config import AgentConfig, ProxyConfig, SessionConfig
from janus.mcp.proxy import JanusMCPProxy
from janus.risk.engine import RiskEngine
from janus.storage.database import DatabaseManager
from janus.storage.session_store import InMemorySessionStore
from janus.tier import current_tier
from janus.web.agent import ChatAgent
from janus.web.events import EventBroadcaster
from janus.web.mcp_chat_agent import McpChatAgent
from janus.web.mock_upstream import MockUpstreamManager
from janus.web.schemas import (
    ChatRequest,
    ChatResponseOut,
    SessionCreateRequest,
    SessionOut,
    ToolCallOut,
)

logger = structlog.get_logger()

# ── Agent definitions ──────────────────────────────────────────────────────────

TESTBED_AGENTS = [
    {
        "agent_id": "free-sdk",
        "name": "Free SDK Agent",
        "role": AgentRole.RESEARCH,
        "tier": "free",
        "integration": "sdk",
        "permissions": ["read_*", "search_*", "api_call"],
    },
    {
        "agent_id": "free-mcp",
        "name": "Free MCP Agent",
        "role": AgentRole.CODE,
        "tier": "free",
        "integration": "mcp",
        "permissions": ["read_*", "write_*", "execute_code", "search_*"],
    },
    {
        "agent_id": "pro-sdk",
        "name": "Pro SDK Agent",
        "role": AgentRole.FINANCIAL,
        "tier": "pro",
        "integration": "sdk",
        "permissions": ["read_*", "database_query", "financial_transfer", "api_call"],
    },
    {
        "agent_id": "pro-mcp",
        "name": "Pro MCP Agent",
        "role": AgentRole.ADMIN,
        "tier": "pro",
        "integration": "mcp",
        "permissions": ["*"],
    },
]

# Map agent_id to tier/integration for routing
AGENT_META: dict[str, dict[str, str]] = {
    a["agent_id"]: {"tier": a["tier"], "integration": a["integration"]}
    for a in TESTBED_AGENTS
}


class TestbedState:
    """Shared testbed application state."""

    def __init__(self) -> None:
        self.free_guardian: Guardian | None = None
        self.pro_guardian: Guardian | None = None
        self.free_proxy: JanusMCPProxy | None = None
        self.pro_proxy: JanusMCPProxy | None = None
        self.free_registry: AgentRegistry | None = None
        self.pro_registry: AgentRegistry | None = None
        self.free_risk_engine: RiskEngine | None = None
        self.pro_risk_engine: RiskEngine | None = None
        self.db: DatabaseManager | None = None
        self.broadcaster = EventBroadcaster()
        self.chat_agents: dict[str, ChatAgent | McpChatAgent] = {}
        self.sessions: dict[str, dict[str, str]] = {}


state = TestbedState()


async def _setup() -> None:
    """Initialize FREE and PRO Guardians, proxies, and register agents."""
    config = JanusConfig()
    # Testbed uses in-memory DB intentionally — ephemeral per session
    db = DatabaseManager(":memory:")
    await db.connect()
    await db.apply_migrations()

    # We need separate registries and session stores for free/pro
    # so they don't share state
    free_registry = AgentRegistry(db)
    free_session_store = InMemorySessionStore()
    free_risk_engine = RiskEngine(free_session_store)

    pro_db = DatabaseManager(":memory:")
    await pro_db.connect()
    await pro_db.apply_migrations()
    pro_registry = AgentRegistry(pro_db)
    pro_session_store = InMemorySessionStore()
    pro_risk_engine = RiskEngine(pro_session_store)

    # LLM components (shared, if API key available)
    classifier = None
    drift_detector = None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        llm_client = AnthropicClientWrapper(api_key=api_key)
        classifier = SecurityClassifier(client=llm_client, config=config.guardian_model)
        drift_detector = SemanticDriftDetector(
            classifier=classifier,
            threshold=config.drift.threshold,
            max_risk_contribution=config.drift.max_risk_contribution,
        )

    # ── 1. Build FREE Guardian (tier checks frozen at construction) ──
    current_tier.reset()  # ensure FREE
    free_guardian = Guardian(
        config=config,
        registry=free_registry,
        risk_engine=free_risk_engine,
        classifier=classifier,
        drift_detector=drift_detector,
    )

    # ── 2. Build FREE MCP Proxy ──
    free_proxy_config = ProxyConfig(
        server_name="janus-testbed-free",
        agent=AgentConfig(
            agent_id="free-mcp",
            name="Free MCP Agent",
            role="code",
            permissions=["read_*", "write_*", "execute_code", "search_*"],
            original_goal="",
        ),
        session=SessionConfig(session_id_prefix="free-mcp"),
    )
    free_proxy = JanusMCPProxy(free_proxy_config)
    free_proxy._guardian = free_guardian
    free_proxy._upstream = MockUpstreamManager()

    # ── 3. Build PRO Guardian ──
    current_tier.activate(generate_license(tier="pro", customer_id="testbed", expiry_days=36500))
    pro_guardian = Guardian(
        config=config,
        registry=pro_registry,
        risk_engine=pro_risk_engine,
        classifier=classifier,
        drift_detector=drift_detector,
    )

    # ── 4. Build PRO MCP Proxy ──
    pro_proxy_config = ProxyConfig(
        server_name="janus-testbed-pro",
        agent=AgentConfig(
            agent_id="pro-mcp",
            name="Pro MCP Agent",
            role="admin",
            permissions=["*"],
            original_goal="",
        ),
        session=SessionConfig(session_id_prefix="pro-mcp"),
    )
    pro_proxy = JanusMCPProxy(pro_proxy_config)
    pro_proxy._guardian = pro_guardian
    pro_proxy._upstream = MockUpstreamManager()

    # Reset tier after construction (pipeline checks already frozen)
    current_tier.reset()

    # ── Register agents in their respective registries ──
    for agent_def in TESTBED_AGENTS:
        identity = AgentIdentity(
            agent_id=agent_def["agent_id"],
            name=agent_def["name"],
            role=agent_def["role"],
            permissions=[ToolPermission(tool_pattern=p) for p in agent_def["permissions"]],
        )
        if agent_def["tier"] == "free":
            await free_registry.register_agent(identity)
        else:
            await pro_registry.register_agent(identity)

    state.free_guardian = free_guardian
    state.pro_guardian = pro_guardian
    state.free_proxy = free_proxy
    state.pro_proxy = pro_proxy
    state.free_registry = free_registry
    state.pro_registry = pro_registry
    state.free_risk_engine = free_risk_engine
    state.pro_risk_engine = pro_risk_engine
    state.db = db

    logger.info(
        "testbed_ready",
        free_pipeline_checks=len(free_guardian._pipeline._checks),
        pro_pipeline_checks=len(pro_guardian._pipeline._checks),
    )


async def _teardown() -> None:
    if state.db:
        await state.db.close()


def _get_guardian(agent_id: str) -> Guardian:
    meta = AGENT_META.get(agent_id, {})
    if meta.get("tier") == "pro":
        assert state.pro_guardian is not None
        return state.pro_guardian
    assert state.free_guardian is not None
    return state.free_guardian


def _get_risk_engine(agent_id: str) -> RiskEngine:
    meta = AGENT_META.get(agent_id, {})
    if meta.get("tier") == "pro":
        assert state.pro_risk_engine is not None
        return state.pro_risk_engine
    assert state.free_risk_engine is not None
    return state.free_risk_engine


def _get_proxy(agent_id: str) -> JanusMCPProxy:
    meta = AGENT_META.get(agent_id, {})
    if meta.get("tier") == "pro":
        assert state.pro_proxy is not None
        return state.pro_proxy
    assert state.free_proxy is not None
    return state.free_proxy


def _get_registry(agent_id: str) -> AgentRegistry:
    meta = AGENT_META.get(agent_id, {})
    if meta.get("tier") == "pro":
        assert state.pro_registry is not None
        return state.pro_registry
    assert state.free_registry is not None
    return state.free_registry


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        await _setup()
        yield
        await _teardown()

    app = FastAPI(
        title="Janus Testbed",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/agents")
    async def list_agents() -> list[dict[str, Any]]:
        # Build pipeline check info per tier
        free_checks = (
            [c.name for c in state.free_guardian._pipeline._checks]
            if state.free_guardian else []
        )
        pro_checks = (
            [c.name for c in state.pro_guardian._pipeline._checks]
            if state.pro_guardian else []
        )
        pro_only = [c for c in pro_checks if c not in free_checks]

        result = []
        for agent_def in TESTBED_AGENTS:
            checks = pro_checks if agent_def["tier"] == "pro" else free_checks
            result.append({
                "agent_id": agent_def["agent_id"],
                "name": agent_def["name"],
                "role": agent_def["role"].value,
                "permissions": agent_def["permissions"],
                "is_locked": False,
                "tier": agent_def["tier"],
                "integration": agent_def["integration"],
                "pipeline_checks": checks,
                "pipeline_check_count": len(checks),
                "pro_only_checks": pro_only,
            })
        return result

    @app.post("/api/sessions", response_model=SessionOut)
    async def create_session(req: SessionCreateRequest) -> SessionOut:
        agent_id = req.agent_id
        meta = AGENT_META.get(agent_id)
        if not meta:
            agent_id = "free-sdk"
            meta = AGENT_META[agent_id]

        guardian = _get_guardian(agent_id)
        _get_risk_engine(agent_id)  # ensure risk engine is initialized
        session_id = f"session-{uuid.uuid4().hex[:8]}"

        state.sessions[session_id] = {
            "agent_id": agent_id,
            "original_goal": req.original_goal,
        }

        if meta["integration"] == "mcp":
            proxy = _get_proxy(agent_id)
            # Set the proxy's session_id for this session
            proxy._session_id = session_id
            state.chat_agents[session_id] = McpChatAgent(
                guardian=guardian,
                proxy=proxy,
                broadcaster=state.broadcaster,
                agent_id=agent_id,
                session_id=session_id,
                original_goal=req.original_goal,
            )
        else:
            state.chat_agents[session_id] = ChatAgent(
                guardian=guardian,
                broadcaster=state.broadcaster,
                agent_id=agent_id,
                session_id=session_id,
                original_goal=req.original_goal,
            )

        return SessionOut(
            session_id=session_id,
            agent_id=agent_id,
            original_goal=req.original_goal,
            risk_score=0.0,
        )

    @app.post("/api/chat", response_model=ChatResponseOut)
    async def chat(req: ChatRequest) -> ChatResponseOut:
        agent = state.chat_agents.get(req.session_id)
        if agent is None:
            # Fallback — create free-sdk agent
            assert state.free_guardian is not None
            agent = ChatAgent(
                guardian=state.free_guardian,
                broadcaster=state.broadcaster,
                agent_id="free-sdk",
                session_id=req.session_id,
            )
            state.chat_agents[req.session_id] = agent
            state.sessions[req.session_id] = {
                "agent_id": "free-sdk",
                "original_goal": req.message,
            }

        response = await agent.chat(req.message)

        return ChatResponseOut(
            message=response.message,
            tool_calls=[
                ToolCallOut(
                    tool_name=tc.tool_name,
                    tool_input=tc.tool_input,
                    verdict=tc.verdict,
                    risk_score=tc.risk_score,
                    risk_delta=tc.risk_delta,
                    result=tc.result,
                    reasons=tc.reasons,
                )
                for tc in response.tool_calls
            ],
            session_id=req.session_id,
        )

    @app.get("/api/sessions/{session_id}/proof")
    async def get_proof_chain(session_id: str) -> list[dict[str, Any]]:
        # Determine which guardian's proof chain to use
        session_meta = state.sessions.get(session_id, {})
        agent_id = session_meta.get("agent_id", "free-sdk")
        guardian = _get_guardian(agent_id)
        chain = guardian.proof_chain.get_chain(session_id)
        return [
            {
                "node_id": n.node_id,
                "parent_hash": n.parent_hash,
                "step": n.step,
                "timestamp": n.timestamp,
                "session_id": n.session_id,
                "agent_id": n.agent_id,
                "tool_name": n.tool_name,
                "verdict": n.verdict,
                "risk_score": n.risk_score,
                "risk_delta": n.risk_delta,
                "content_hash": n.content_hash,
            }
            for n in chain
        ]

    @app.post("/api/sessions/{session_id}/proof/verify")
    async def verify_proof_chain(session_id: str) -> dict[str, Any]:
        session_meta = state.sessions.get(session_id, {})
        agent_id = session_meta.get("agent_id", "free-sdk")
        guardian = _get_guardian(agent_id)
        valid = guardian.proof_chain.verify(session_id)
        chain_len = len(guardian.proof_chain.get_chain(session_id))
        return {
            "valid": valid,
            "chain_length": chain_len,
            "session_id": session_id,
        }

    @app.get("/api/threat-intel")
    async def get_threat_intel() -> list[dict[str, Any]]:
        # Combine threat intel from both guardians
        all_patterns = []
        for guardian in [state.free_guardian, state.pro_guardian]:
            if guardian:
                patterns = guardian.threat_intel_db.get_all_patterns()
                for p in patterns:
                    all_patterns.append({
                        "pattern_id": p.pattern_id,
                        "pattern_type": p.pattern_type,
                        "tool_sequence": p.tool_sequence,
                        "risk_contribution": p.risk_contribution,
                        "confidence": p.confidence,
                        "first_seen": p.first_seen.isoformat(),
                        "times_seen": p.times_seen,
                        "source": p.source,
                    })
        # Deduplicate by pattern_id
        seen: set[str] = set()
        unique = []
        for p in all_patterns:
            if p["pattern_id"] not in seen:
                seen.add(p["pattern_id"])
                unique.append(p)
        return unique

    @app.get("/api/threat-intel/stats")
    async def get_threat_intel_stats() -> dict[str, Any]:
        # Use pro guardian stats if available (superset)
        guardian = state.pro_guardian or state.free_guardian
        assert guardian is not None
        return guardian.threat_intel_db.get_stats()

    @app.websocket("/api/ws/session/{session_id}")
    async def websocket_session(
        websocket: WebSocket, session_id: str,
    ) -> None:
        await websocket.accept()
        try:
            async for event in state.broadcaster.subscribe(session_id):
                await websocket.send_json(event.to_dict())
        except WebSocketDisconnect:
            pass

    return app


def run_server(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Run the testbed server with uvicorn."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
