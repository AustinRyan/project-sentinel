"""FastAPI application for the Sentinel demo UI backend."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from sentinel.config import SentinelConfig
from sentinel.core.guardian import Guardian
from sentinel.identity.agent import AgentIdentity, AgentRole, ToolPermission
from sentinel.identity.registry import AgentRegistry
from sentinel.risk.engine import RiskEngine
from sentinel.storage.database import DatabaseManager
from sentinel.storage.session_store import InMemorySessionStore
from sentinel.web.agent import ChatAgent
from sentinel.web.events import EventBroadcaster
from sentinel.web.schemas import (
    AgentOut,
    ChatRequest,
    ChatResponseOut,
    HealthOut,
    SessionCreateRequest,
    SessionOut,
    ToolCallOut,
)

logger = structlog.get_logger()


class AppState:
    """Shared application state."""

    def __init__(self) -> None:
        self.guardian: Guardian | None = None
        self.registry: AgentRegistry | None = None
        self.risk_engine: RiskEngine | None = None
        self.session_store: InMemorySessionStore | None = None
        self.db: DatabaseManager | None = None
        self.broadcaster = EventBroadcaster()
        self.chat_agents: dict[str, ChatAgent] = {}
        self.sessions: dict[str, dict[str, str]] = {}


state = AppState()


async def _setup() -> None:
    """Initialize all Sentinel components."""
    config = SentinelConfig()
    db = DatabaseManager(":memory:")
    await db.connect()
    await db.apply_migrations()

    registry = AgentRegistry(db)
    session_store = InMemorySessionStore()
    risk_engine = RiskEngine(session_store)

    guardian = Guardian(
        config=config,
        registry=registry,
        risk_engine=risk_engine,
    )

    demo_agent = AgentIdentity(
        agent_id="demo-agent",
        name="Demo Research Bot",
        role=AgentRole.RESEARCH,
        permissions=[
            ToolPermission(tool_pattern="read_*"),
            ToolPermission(tool_pattern="search_*"),
            ToolPermission(tool_pattern="api_call"),
            ToolPermission(tool_pattern="execute_code"),
            ToolPermission(tool_pattern="write_file"),
            ToolPermission(tool_pattern="database_query"),
        ],
    )
    await registry.register_agent(demo_agent)

    state.guardian = guardian
    state.registry = registry
    state.risk_engine = risk_engine
    state.session_store = session_store
    state.db = db


async def _teardown() -> None:
    if state.db:
        await state.db.close()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        await _setup()
        yield
        await _teardown()

    app = FastAPI(
        title="Sentinel Security Dashboard",
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

    @app.get("/api/health", response_model=HealthOut)
    async def health() -> HealthOut:
        assert state.guardian is not None
        metrics = state.guardian.health.get_metrics()
        return HealthOut(
            status="ok",
            total_requests=metrics.total_requests,
            error_rate=metrics.error_rate,
            circuit_breaker=state.guardian.circuit_breaker.state.value,
        )

    @app.get("/api/sessions", response_model=list[SessionOut])
    async def list_sessions() -> list[SessionOut]:
        assert state.session_store is not None
        assert state.risk_engine is not None
        result = []
        for sid, meta in state.sessions.items():
            result.append(SessionOut(
                session_id=sid,
                agent_id=meta.get("agent_id", ""),
                original_goal=meta.get("original_goal", ""),
                risk_score=state.risk_engine.get_score(sid),
            ))
        return result

    @app.post("/api/sessions", response_model=SessionOut)
    async def create_session(req: SessionCreateRequest) -> SessionOut:
        assert state.guardian is not None
        assert state.risk_engine is not None
        session_id = f"session-{uuid.uuid4().hex[:8]}"
        state.sessions[session_id] = {
            "agent_id": req.agent_id,
            "original_goal": req.original_goal,
        }
        state.chat_agents[session_id] = ChatAgent(
            guardian=state.guardian,
            broadcaster=state.broadcaster,
            agent_id=req.agent_id,
            session_id=session_id,
            original_goal=req.original_goal,
        )
        return SessionOut(
            session_id=session_id,
            agent_id=req.agent_id,
            original_goal=req.original_goal,
            risk_score=0.0,
        )

    @app.post("/api/chat", response_model=ChatResponseOut)
    async def chat(req: ChatRequest) -> ChatResponseOut:
        agent = state.chat_agents.get(req.session_id)
        if agent is None:
            assert state.guardian is not None
            agent = ChatAgent(
                guardian=state.guardian,
                broadcaster=state.broadcaster,
                agent_id="demo-agent",
                session_id=req.session_id,
            )
            state.chat_agents[req.session_id] = agent
            state.sessions[req.session_id] = {
                "agent_id": "demo-agent",
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

    @app.get("/api/agents", response_model=list[AgentOut])
    async def list_agents() -> list[AgentOut]:
        assert state.registry is not None
        agents = await state.registry.list_agents()
        return [
            AgentOut(
                agent_id=a.agent_id,
                name=a.name,
                role=a.role.value,
                permissions=[p.tool_pattern for p in a.permissions],
                is_locked=a.is_locked,
            )
            for a in agents
        ]

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


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server with uvicorn."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)
