"""FastAPI application for the Janus demo UI backend."""
from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import APIRouter, Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from janus.config import JanusConfig
from janus.core.approval import ApprovalManager, needs_human_review
from janus.core.guardian import Guardian
from janus.drift.detector import SemanticDriftDetector
from janus.exporters.coordinator import ExporterCoordinator
from janus.forensics.explainer import TraceExplainer
from janus.forensics.recorder import BlackBoxRecorder
from janus.identity.agent import AgentIdentity, AgentRole, ToolPermission
from janus.identity.registry import AgentRegistry
from janus.licensing import _reset_verification_key, generate_license
from janus.llm.classifier import SecurityClassifier
from janus.risk.engine import RiskEngine
from janus.storage.database import DatabaseManager
from janus.storage.persistent_session_store import PersistentSessionStore
from janus.storage.session_store import InMemorySessionStore
from janus.tier import current_tier
from janus.web.agent import ChatAgent
from janus.web.auth import RateLimitMiddleware, require_api_key, require_pro_tier
from janus.web.events import EventBroadcaster
from janus.web.schemas import (
    AgentOut,
    ApprovalDecisionOut,
    ApprovalDecisionRequest,
    ApprovalRequestOut,
    ChatRequest,
    ChatResponseOut,
    HealthFullOut,
    HealthOut,
    MessageOut,
    RiskEventOut,
    SessionCreateRequest,
    SessionOut,
    TaintEntryOut,
    ToolCallOut,
    ToolEvalRequest,
    ToolEvalResponse,
    TraceOut,
)

logger = structlog.get_logger()


class AppState:
    """Shared application state."""

    def __init__(self) -> None:
        self.guardian: Guardian | None = None
        self.registry: AgentRegistry | None = None
        self.risk_engine: RiskEngine | None = None
        self.session_store: PersistentSessionStore | InMemorySessionStore | None = None
        self.db: DatabaseManager | None = None
        self.broadcaster = EventBroadcaster()
        self.chat_agents: dict[str, ChatAgent] = {}
        self.sessions: dict[str, dict[str, str]] = {}
        self.recorder: BlackBoxRecorder | None = None
        self.exporter_coordinator: ExporterCoordinator | None = None
        self.approval_manager: ApprovalManager | None = None
        self.tool_registry: Any | None = None
        self.tool_executor: Any | None = None
        self.api_key: str = ""


state = AppState()


async def _setup() -> None:
    """Initialize all Janus components."""
    # Load full config from TOML if path is set
    config_path = os.environ.get("JANUS_CONFIG_PATH")
    if config_path:
        config = JanusConfig.from_toml(config_path)
    else:
        config = JanusConfig()

    # Apply configurable policies
    from janus.risk import thresholds
    thresholds.configure(config.risk, config.policy)

    # Activate license BEFORE Guardian construction (tier checks at init)
    if config.license_key:
        current_tier.activate(config.license_key)
    elif os.environ.get("JANUS_DEV_MODE", "").lower() == "true":
        # Auto-activate PRO in dev mode — ensure a signing key is available
        if not os.environ.get("JANUS_LICENSE_SECRET"):
            os.environ["JANUS_LICENSE_SECRET"] = "janus-dev-mode-key"
            _reset_verification_key()
        logger.warning(
            "JANUS_DEV_MODE is active — PRO features enabled. Do NOT use in production."
        )
        current_tier.activate(generate_license(tier="pro", customer_id="dashboard", expiry_days=36500))

    db_path = os.environ.get("JANUS_DB_PATH", config.database_path)
    db = DatabaseManager(db_path)
    await db.connect()
    await db.apply_migrations()

    registry = AgentRegistry(db)
    session_store = PersistentSessionStore(db)
    await session_store.initialize()
    risk_engine = RiskEngine(session_store)

    # LLM-based security: gracefully degrade if no API key
    classifier = None
    drift_detector = None
    api_key = config.guardian_model.api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
    provider_type = config.guardian_model.provider
    if api_key or provider_type == "ollama":
        from janus.llm.providers import create_provider

        llm_client = create_provider(
            provider=provider_type,
            api_key=api_key,
            base_url=config.guardian_model.base_url,
        )
        classifier = SecurityClassifier(client=llm_client, config=config.guardian_model)
        drift_detector = SemanticDriftDetector(
            classifier=classifier,
            threshold=config.drift.threshold,
            max_risk_contribution=config.drift.max_risk_contribution,
        )

    # Create BlackBoxRecorder for persistent traces
    explainer = TraceExplainer(classifier=classifier) if classifier else TraceExplainer()
    recorder = BlackBoxRecorder(db, explainer)

    guardian = Guardian(
        config=config,
        registry=registry,
        risk_engine=risk_engine,
        classifier=classifier,
        drift_detector=drift_detector,
        recorder=recorder,
    )

    # Create exporter coordinator
    exporter_coordinator = ExporterCoordinator(config.exporters)

    # Register agent personas
    agents = [
        AgentIdentity(
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
        ),
        AgentIdentity(
            agent_id="marketing-bot",
            name="Marketing Bot",
            role=AgentRole.COMMUNICATION,
            permissions=[
                ToolPermission(tool_pattern="search_*"),
                ToolPermission(tool_pattern="read_*"),
                ToolPermission(tool_pattern="send_email"),
                ToolPermission(tool_pattern="send_message"),
            ],
        ),
        AgentIdentity(
            agent_id="developer-bot",
            name="Developer Bot",
            role=AgentRole.CODE,
            permissions=[
                ToolPermission(tool_pattern="read_*"),
                ToolPermission(tool_pattern="write_*"),
                ToolPermission(tool_pattern="execute_code"),
                ToolPermission(tool_pattern="search_*"),
                ToolPermission(tool_pattern="database_query"),
            ],
        ),
        AgentIdentity(
            agent_id="finance-bot",
            name="Finance Bot",
            role=AgentRole.FINANCIAL,
            permissions=[
                ToolPermission(tool_pattern="read_*"),
                ToolPermission(tool_pattern="database_query"),
                ToolPermission(tool_pattern="financial_transfer"),
                ToolPermission(tool_pattern="api_call"),
            ],
        ),
        AgentIdentity(
            agent_id="research-bot",
            name="Research Bot",
            role=AgentRole.RESEARCH,
            permissions=[
                ToolPermission(tool_pattern="read_*"),
                ToolPermission(tool_pattern="search_*"),
                ToolPermission(tool_pattern="api_call"),
            ],
        ),
        AgentIdentity(
            agent_id="admin-bot",
            name="Admin Bot",
            role=AgentRole.ADMIN,
            permissions=[
                ToolPermission(tool_pattern="*"),
            ],
        ),
    ]
    for agent in agents:
        try:
            await registry.register_agent(agent)
        except Exception:
            pass  # Agent already exists from previous run

    # Create tool registry and executor
    from janus.tools.executor import ToolExecutor
    from janus.tools.registry import ToolRegistry

    tool_registry = ToolRegistry(db)
    tool_executor = ToolExecutor(registry=tool_registry)
    await tool_executor.refresh_definitions()

    # Create approval manager for HITL workflow
    approval_manager = ApprovalManager(
        db=db,
        broadcaster=state.broadcaster,
        tool_executor=tool_executor,
        guardian=guardian,
        exporter_coordinator=exporter_coordinator,
    )

    state.guardian = guardian
    state.registry = registry
    state.risk_engine = risk_engine
    state.session_store = session_store
    state.db = db
    state.recorder = recorder
    state.exporter_coordinator = exporter_coordinator
    state.approval_manager = approval_manager
    state.tool_registry = tool_registry
    state.tool_executor = tool_executor
    state.api_key = api_key

    # Restore active sessions from DB so they survive restarts
    if isinstance(session_store, PersistentSessionStore):
        for meta in await session_store.get_all_session_metadata():
            sid = str(meta["session_id"])
            state.sessions[sid] = {
                "agent_id": str(meta["agent_id"] or ""),
                "original_goal": str(meta["original_goal"] or ""),
            }


async def _teardown() -> None:
    if isinstance(state.session_store, PersistentSessionStore):
        await state.session_store.shutdown()
    if state.db:
        await state.db.close()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        await _setup()
        yield
        await _teardown()

    app = FastAPI(
        title="Janus Security Dashboard",
        version="0.2.0",
        lifespan=lifespan,
    )

    # CORS: read allowed origins from env var (comma-separated), default to localhost
    cors_raw = os.environ.get("JANUS_CORS_ORIGINS", "")
    allowed_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
    if not allowed_origins:
        allowed_origins = ["http://localhost:3000", "http://localhost:8000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RateLimitMiddleware)

    # Health endpoint — no auth required
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

    # Protected endpoints — require API key when JANUS_API_KEY is set
    api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

    @api.get("/sessions", response_model=list[SessionOut])
    async def list_sessions() -> list[SessionOut]:
        assert state.session_store is not None
        assert state.risk_engine is not None
        # Query from DB if persistent store, otherwise use in-memory
        if isinstance(state.session_store, PersistentSessionStore):
            all_meta = await state.session_store.get_all_session_metadata()
            result = []
            for meta in all_meta:
                sid = str(meta["session_id"])
                result.append(SessionOut(
                    session_id=sid,
                    agent_id=str(meta["agent_id"]),
                    original_goal=str(meta["original_goal"]),
                    risk_score=float(meta.get("risk_score", 0.0)),
                ))
            return result
        result = []
        for sid, meta in state.sessions.items():
            result.append(SessionOut(
                session_id=sid,
                agent_id=meta.get("agent_id", ""),
                original_goal=meta.get("original_goal", ""),
                risk_score=state.risk_engine.get_score(sid),
            ))
        return result

    @api.post("/sessions", response_model=SessionOut)
    async def create_session(req: SessionCreateRequest) -> SessionOut:
        assert state.guardian is not None
        assert state.risk_engine is not None
        assert state.session_store is not None
        session_id = f"session-{uuid.uuid4().hex[:8]}"
        state.sessions[session_id] = {
            "agent_id": req.agent_id,
            "original_goal": req.original_goal,
        }
        # Ensure session exists in the persistent store and set agent_id
        state.session_store.get_or_create_session(session_id)
        if req.original_goal:
            state.session_store.set_goal(session_id, req.original_goal)
        if isinstance(state.session_store, PersistentSessionStore):
            await state.session_store.set_agent_id(session_id, req.agent_id)

        state.chat_agents[session_id] = ChatAgent(
            guardian=state.guardian,
            broadcaster=state.broadcaster,
            agent_id=req.agent_id,
            session_id=session_id,
            original_goal=req.original_goal,
            api_key=state.api_key,
            exporter_coordinator=state.exporter_coordinator,
            approval_manager=state.approval_manager,
            tool_executor=state.tool_executor,
        )
        return SessionOut(
            session_id=session_id,
            agent_id=req.agent_id,
            original_goal=req.original_goal,
            risk_score=0.0,
        )

    @api.post("/chat", response_model=ChatResponseOut)
    async def chat(req: ChatRequest) -> ChatResponseOut:
        agent = state.chat_agents.get(req.session_id)
        if agent is None:
            assert state.guardian is not None
            # Restore from session metadata if available
            meta = state.sessions.get(req.session_id, {})
            agent_id = meta.get("agent_id", "") or "demo-agent"
            original_goal = meta.get("original_goal", "") or req.message

            agent = ChatAgent(
                guardian=state.guardian,
                broadcaster=state.broadcaster,
                agent_id=agent_id,
                session_id=req.session_id,
                original_goal=original_goal,
                api_key=state.api_key,
                exporter_coordinator=state.exporter_coordinator,
                approval_manager=state.approval_manager,
                tool_executor=state.tool_executor,
            )
            # Restore conversation history from DB
            if state.db is not None:
                rows = await state.db.fetchall(
                    "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id",
                    (req.session_id,),
                )
                history = [{"role": row["role"], "content": row["content"]} for row in rows]
                agent.set_history(history)

            state.chat_agents[req.session_id] = agent
            if req.session_id not in state.sessions:
                state.sessions[req.session_id] = {
                    "agent_id": agent_id,
                    "original_goal": req.message,
                }

        response = await agent.chat(req.message)

        # Persist messages to DB
        if state.db is not None:
            tool_calls_data = [
                {
                    "tool_name": tc.tool_name,
                    "tool_input": tc.tool_input,
                    "verdict": tc.verdict,
                    "risk_score": tc.risk_score,
                    "risk_delta": tc.risk_delta,
                    "reasons": tc.reasons,
                }
                for tc in response.tool_calls
            ]
            await state.db.execute(
                "INSERT INTO chat_messages (session_id, role, content, tool_calls_json) VALUES (?, ?, ?, ?)",
                (req.session_id, "user", req.message, "[]"),
            )
            await state.db.execute(
                "INSERT INTO chat_messages (session_id, role, content, tool_calls_json) VALUES (?, ?, ?, ?)",
                (req.session_id, "assistant", response.message, json.dumps(tool_calls_data, default=str)),
            )
            await state.db.commit()

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

    @api.get("/agents", response_model=list[AgentOut])
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

    @api.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
    async def get_session_messages(session_id: str) -> list[MessageOut]:
        assert state.db is not None
        rows = await state.db.fetchall(
            "SELECT role, content, tool_calls_json FROM chat_messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        result = []
        for row in rows:
            tool_calls_raw = json.loads(row["tool_calls_json"]) if row["tool_calls_json"] else []
            tool_calls = [
                ToolCallOut(
                    tool_name=tc.get("tool_name", ""),
                    tool_input=tc.get("tool_input", {}),
                    verdict=tc.get("verdict", ""),
                    risk_score=tc.get("risk_score", 0.0),
                    risk_delta=tc.get("risk_delta", 0.0),
                    reasons=tc.get("reasons", []),
                )
                for tc in tool_calls_raw
            ]
            result.append(MessageOut(
                role=row["role"],
                content=row["content"],
                tool_calls=tool_calls,
            ))
        return result

    @api.get("/sessions/{session_id}/proof")
    async def get_proof_chain(session_id: str) -> list[dict[str, Any]]:
        assert state.guardian is not None
        chain = state.guardian.proof_chain.get_chain(
            session_id
        )
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

    @api.post("/sessions/{session_id}/proof/verify")
    async def verify_proof_chain(
        session_id: str,
    ) -> dict[str, Any]:
        assert state.guardian is not None
        valid = state.guardian.proof_chain.verify(
            session_id
        )
        chain_len = len(
            state.guardian.proof_chain.get_chain(
                session_id
            )
        )
        return {
            "valid": valid,
            "chain_length": chain_len,
            "session_id": session_id,
        }

    @api.get("/threat-intel", dependencies=[Depends(require_pro_tier)])
    async def get_threat_intel() -> list[dict[str, Any]]:
        assert state.guardian is not None
        patterns = (
            state.guardian.threat_intel_db
            .get_all_patterns()
        )
        return [
            {
                "pattern_id": p.pattern_id,
                "pattern_type": p.pattern_type,
                "tool_sequence": p.tool_sequence,
                "risk_contribution": p.risk_contribution,
                "confidence": p.confidence,
                "first_seen": p.first_seen.isoformat(),
                "times_seen": p.times_seen,
                "source": p.source,
            }
            for p in patterns
        ]

    @api.get("/threat-intel/stats", dependencies=[Depends(require_pro_tier)])
    async def get_threat_intel_stats() -> dict[str, Any]:
        assert state.guardian is not None
        return (
            state.guardian.threat_intel_db.get_stats()
        )

    @api.get("/traces", response_model=list[TraceOut])
    async def list_traces(
        session_id: str | None = None,
        verdict: str | None = None,
        limit: int = 50,
    ) -> list[TraceOut]:
        assert state.recorder is not None
        if session_id:
            traces = await state.recorder.get_traces_by_session(session_id)
        elif verdict:
            traces = await state.recorder.get_traces_by_verdict(verdict, limit=limit)
        else:
            traces = await state.recorder.get_recent_traces(limit=limit)
        return [
            TraceOut(
                trace_id=t.trace_id,
                session_id=t.session_id,
                agent_id=t.agent_id,
                tool_name=t.tool_name,
                verdict=t.verdict,
                risk_score=t.risk_score,
                risk_delta=t.risk_delta,
                explanation=t.explanation,
                timestamp=t.timestamp.isoformat(),
                reasons=t.reasons,
            )
            for t in traces
        ]

    @api.get("/export/traces", dependencies=[Depends(require_pro_tier)])
    async def export_traces(
        format: str = "json",
        verdict: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        min_risk: float | None = None,
        limit: int = 10000,
    ) -> Any:
        assert state.db is not None
        from fastapi.responses import Response

        from janus.forensics.exporter import TraceExporter

        exporter = TraceExporter(state.db)
        traces = await exporter.query_traces(
            date_from=date_from,
            date_to=date_to,
            verdict=verdict,
            agent_id=agent_id,
            session_id=session_id,
            min_risk=min_risk,
            limit=limit,
        )

        if format == "csv":
            content = exporter.to_csv(traces)
            media_type = "text/csv"
            ext = "csv"
        elif format == "jsonl":
            content = exporter.to_jsonl(traces)
            media_type = "application/x-ndjson"
            ext = "jsonl"
        else:
            content = exporter.to_json(traces)
            media_type = "application/json"
            ext = "json"

        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=traces.{ext}"},
        )

    # ── Direct tool-call evaluation (no LLM needed) ──────────────────────

    @api.post("/evaluate", response_model=ToolEvalResponse)
    async def evaluate_tool_call(req: ToolEvalRequest) -> ToolEvalResponse:
        """Evaluate a tool call through the full Guardian pipeline without LLM chat.

        Use this to test security decisions directly.
        """
        assert state.guardian is not None
        assert state.session_store is not None

        # Ensure session exists
        if req.session_id not in state.sessions:
            state.sessions[req.session_id] = {
                "agent_id": req.agent_id,
                "original_goal": req.original_goal,
            }
            state.session_store.get_or_create_session(req.session_id)
            if req.original_goal:
                state.session_store.set_goal(req.session_id, req.original_goal)
            if isinstance(state.session_store, PersistentSessionStore):
                await state.session_store.set_agent_id(req.session_id, req.agent_id)

        verdict = await state.guardian.wrap_tool_call(
            agent_id=req.agent_id,
            session_id=req.session_id,
            original_goal=req.original_goal or state.sessions[req.session_id].get("original_goal", ""),
            tool_name=req.tool_name,
            tool_input=req.tool_input,
        )

        # Broadcast as real-time event so the dashboard sees it
        from janus.web.events import SecurityEvent
        await state.broadcaster.publish(SecurityEvent(
            event_type="verdict",
            session_id=req.session_id,
            data={
                "verdict": verdict.verdict.value,
                "risk_score": verdict.risk_score,
                "risk_delta": verdict.risk_delta,
                "tool_name": req.tool_name,
                "tool_input": req.tool_input,
                "reasons": verdict.reasons,
                "drift_score": getattr(verdict, "drift_score", None),
                "itdr_signals": getattr(verdict, "itdr_signals", []),
                "recommended_action": getattr(verdict, "recommended_action", ""),
                "trace_id": getattr(verdict, "trace_id", ""),
                "check_results": [
                    {
                        "check_name": cr.check_name,
                        "passed": cr.passed,
                        "risk_contribution": cr.risk_contribution,
                        "reason": cr.reason,
                        "metadata": cr.metadata,
                        "force_verdict": cr.force_verdict.value if cr.force_verdict else None,
                    }
                    for cr in verdict.check_results
                ],
            },
        ))

        # Create approval request only for judgment-call blocks (not hard policy violations)
        approval_id: str | None = None
        check_results_dicts = [
            {
                "check_name": cr.check_name,
                "passed": cr.passed,
                "risk_contribution": cr.risk_contribution,
                "reason": cr.reason,
                "metadata": cr.metadata,
                "force_verdict": cr.force_verdict.value if cr.force_verdict else None,
            }
            for cr in verdict.check_results
        ]
        if (
            verdict.verdict.value != "allow"
            and state.approval_manager is not None
            and needs_human_review(verdict.verdict.value, check_results_dicts)
        ):
            approval = await state.approval_manager.create(
                session_id=req.session_id,
                agent_id=req.agent_id,
                tool_name=req.tool_name,
                tool_input=req.tool_input,
                original_goal=req.original_goal or state.sessions[req.session_id].get("original_goal", ""),
                verdict=verdict.verdict.value,
                risk_score=verdict.risk_score,
                risk_delta=verdict.risk_delta,
                reasons=verdict.reasons,
                check_results=check_results_dicts,
                trace_id=getattr(verdict, "trace_id", ""),
            )
            approval_id = approval.id

        return ToolEvalResponse(
            verdict=verdict.verdict.value,
            risk_score=verdict.risk_score,
            risk_delta=verdict.risk_delta,
            reasons=verdict.reasons,
            session_id=req.session_id,
            tool_name=req.tool_name,
            approval_id=approval_id,
        )

    # ── Approval endpoints (HITL) ───────────────────────────────────────

    @api.get("/approvals", response_model=list[ApprovalRequestOut])
    async def list_approvals(
        status: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[ApprovalRequestOut]:
        assert state.approval_manager is not None
        requests = await state.approval_manager.get_all(
            status=status, session_id=session_id, limit=limit
        )
        return [
            ApprovalRequestOut(**r.to_dict())
            for r in requests
        ]

    @api.get("/approvals/stats")
    async def approval_stats() -> dict[str, int]:
        assert state.approval_manager is not None
        return await state.approval_manager.get_stats()

    @api.get("/approvals/{approval_id}", response_model=ApprovalRequestOut)
    async def get_approval(approval_id: str) -> ApprovalRequestOut:
        assert state.approval_manager is not None
        request = await state.approval_manager.get_by_id(approval_id)
        if request is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Approval not found")
        return ApprovalRequestOut(**request.to_dict())

    @api.post("/approvals/{approval_id}/approve", response_model=ApprovalDecisionOut)
    async def approve_request(
        approval_id: str, body: ApprovalDecisionRequest,
    ) -> ApprovalDecisionOut:
        assert state.approval_manager is not None
        result = await state.approval_manager.approve(
            approval_id, decided_by=body.decided_by, reason=body.reason,
        )
        if result is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Approval not found or already resolved")
        return ApprovalDecisionOut(
            id=result.id,
            status=result.status,
            decided_by=result.decided_by,
            decided_at=result.decided_at,
            decision_reason=result.decision_reason,
            tool_result=result.tool_result,
        )

    @api.post("/approvals/{approval_id}/reject", response_model=ApprovalDecisionOut)
    async def reject_request(
        approval_id: str, body: ApprovalDecisionRequest,
    ) -> ApprovalDecisionOut:
        assert state.approval_manager is not None
        result = await state.approval_manager.reject(
            approval_id, decided_by=body.decided_by, reason=body.reason,
        )
        if result is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Approval not found or already resolved")
        return ApprovalDecisionOut(
            id=result.id,
            status=result.status,
            decided_by=result.decided_by,
            decided_at=result.decided_at,
            decision_reason=result.decision_reason,
        )

    # ── Monitor endpoints ────────────────────────────────────────────────

    @api.get("/sessions/{session_id}/events", response_model=list[RiskEventOut])
    async def get_session_events(session_id: str) -> list[RiskEventOut]:
        """Risk event history for a session (powers the risk timeline chart)."""
        assert state.risk_engine is not None
        events = state.risk_engine.session_store.get_events(session_id)
        return [
            RiskEventOut(
                risk_delta=e.risk_delta,
                new_score=e.new_score,
                tool_name=e.tool_name,
                reason=e.reason,
                timestamp=e.timestamp.isoformat(),
            )
            for e in events
        ]

    @api.get("/sessions/{session_id}/taint", response_model=list[TaintEntryOut], dependencies=[Depends(require_pro_tier)])
    async def get_session_taint(session_id: str) -> list[TaintEntryOut]:
        """Active taint entries for a session."""
        assert state.guardian is not None
        entries = state.guardian.taint_tracker.get_active_taints(session_id)
        return [
            TaintEntryOut(
                label=e.label.value,
                source_tool=e.source_tool,
                source_step=e.source_step,
                patterns_matched=e.patterns_matched,
                timestamp=e.timestamp.isoformat(),
            )
            for e in entries
        ]

    @api.get("/health/full", response_model=HealthFullOut)
    async def health_full() -> HealthFullOut:
        """Full health metrics including latency percentiles."""
        assert state.guardian is not None
        metrics = state.guardian.health.get_metrics()
        return HealthFullOut(
            status="ok",
            total_requests=metrics.total_requests,
            successful_requests=metrics.successful_requests,
            failed_requests=metrics.failed_requests,
            avg_latency_ms=metrics.avg_latency_ms,
            p95_latency_ms=metrics.p95_latency_ms,
            error_rate=metrics.error_rate,
            circuit_breaker=state.guardian.circuit_breaker.state.value,
            active_sessions=len(state.sessions),
        )

    app.include_router(api)

    # Tool management routes (auth-protected)
    from janus.web.tool_routes import router as tool_router
    tool_router.dependencies = [Depends(require_api_key)]
    app.include_router(tool_router)

    # Licensing routes — separate from auth-protected router (Stripe signs its own requests)
    from janus.web.licensing_routes import router as licensing_router

    app.include_router(licensing_router)

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

    @app.websocket("/api/ws/monitor")
    async def websocket_monitor(websocket: WebSocket) -> None:
        """Global WebSocket that receives ALL session events for the monitor dashboard."""
        await websocket.accept()
        try:
            async for event in state.broadcaster.subscribe("*"):
                await websocket.send_json(event.to_dict())
        except WebSocketDisconnect:
            pass

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server with uvicorn."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)
