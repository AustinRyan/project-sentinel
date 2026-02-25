"""Janus MCP Proxy -- security-gated proxy for upstream MCP servers."""
from __future__ import annotations

import os
import uuid
from typing import Any

import structlog
from mcp import types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from janus.config import JanusConfig
from janus.core.decision import Verdict
from janus.core.guardian import Guardian
from janus.identity.agent import AgentIdentity, AgentRole, ToolPermission
from janus.identity.registry import AgentRegistry
from janus.mcp.client import UpstreamManager
from janus.mcp.config import ProxyConfig
from janus.risk.engine import RiskEngine
from janus.storage.database import DatabaseManager
from janus.storage.session_store import InMemorySessionStore

logger = structlog.get_logger()


class JanusMCPProxy:
    """MCP proxy server that gates all tool calls through Guardian."""

    def __init__(self, config: ProxyConfig) -> None:
        self._config = config
        self._server = Server(config.server_name)
        self._upstream = UpstreamManager()
        self._guardian: Guardian | None = None
        self._db: DatabaseManager | None = None
        self._session_id: str = ""
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self._server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
        async def handle_list_tools() -> list[types.Tool]:
            return self._upstream.get_all_tools()

        @self._server.call_tool()  # type: ignore[untyped-decorator]
        async def handle_call_tool(
            name: str, arguments: dict[str, Any] | None
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            return await self._intercept_and_forward(name, arguments or {})

    async def setup(self) -> None:
        """Initialize Guardian, connect to upstream servers, register agent."""
        self._db = DatabaseManager(self._config.database_path)
        await self._db.connect()
        await self._db.apply_migrations()

        janus_config = JanusConfig(**self._config.janus)
        registry = AgentRegistry(self._db)
        session_store = InMemorySessionStore()
        risk_engine = RiskEngine(session_store)

        classifier = None
        drift_detector = None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            from janus.drift.detector import SemanticDriftDetector
            from janus.llm.classifier import SecurityClassifier
            from janus.llm.client import AnthropicClientWrapper

            llm_client = AnthropicClientWrapper(api_key=api_key)
            classifier = SecurityClassifier(
                client=llm_client, config=janus_config.guardian_model
            )
            drift_detector = SemanticDriftDetector(
                classifier=classifier,
                threshold=janus_config.drift.threshold,
                max_risk_contribution=janus_config.drift.max_risk_contribution,
            )

        self._guardian = Guardian(
            config=janus_config,
            registry=registry,
            risk_engine=risk_engine,
            classifier=classifier,
            drift_detector=drift_detector,
        )

        agent_cfg = self._config.agent
        agent_role = AgentRole(agent_cfg.role)
        permissions = [ToolPermission(tool_pattern=p) for p in agent_cfg.permissions]
        agent = AgentIdentity(
            agent_id=agent_cfg.agent_id,
            name=agent_cfg.name,
            role=agent_role,
            permissions=permissions,
        )
        await registry.register_agent(agent)

        if self._config.session.persistent_session_id:
            self._session_id = self._config.session.persistent_session_id
        else:
            self._session_id = (
                f"{self._config.session.session_id_prefix}-{uuid.uuid4().hex[:8]}"
            )

        await self._upstream.connect(self._config.upstream_servers)

        logger.info(
            "proxy_ready",
            server_name=self._config.server_name,
            upstream_count=len(self._upstream.server_names),
            tool_count=self._upstream.tool_count,
            session_id=self._session_id,
            agent_id=agent_cfg.agent_id,
        )

    async def _intercept_and_forward(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        assert self._guardian is not None

        verdict = await self._guardian.wrap_tool_call(
            agent_id=self._config.agent.agent_id,
            session_id=self._session_id,
            original_goal=self._config.agent.original_goal,
            tool_name=tool_name,
            tool_input=arguments,
        )

        logger.info(
            "tool_call_intercepted",
            tool=tool_name,
            verdict=verdict.verdict.value,
            risk_score=verdict.risk_score,
            risk_delta=verdict.risk_delta,
        )

        if verdict.verdict == Verdict.ALLOW:
            try:
                result = await self._upstream.call_tool(tool_name, arguments)
                return list(result.content)  # type: ignore[arg-type]
            except Exception as exc:
                logger.error("upstream_call_failed", tool=tool_name, error=str(exc))
                return [types.TextContent(type="text", text=f"Upstream error: {exc}")]

        reasons = "; ".join(verdict.reasons) if verdict.reasons else "No details"
        if verdict.verdict == Verdict.CHALLENGE:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"[JANUS CHALLENGE] Tool '{tool_name}' requires verification. "
                        f"Risk: {verdict.risk_score:.1f}. Reasons: {reasons}. "
                        f"Action: {verdict.recommended_action}"
                    ),
                )
            ]

        return [
            types.TextContent(
                type="text",
                text=(
                    f"[JANUS BLOCKED] Tool '{tool_name}' was blocked by security policy. "
                    f"Verdict: {verdict.verdict.value}. Risk: {verdict.risk_score:.1f}. "
                    f"Reasons: {reasons}. "
                    f"Action: {verdict.recommended_action}"
                ),
            )
        ]

    def get_initialization_options(self) -> InitializationOptions:
        return InitializationOptions(
            server_name=self._config.server_name,
            server_version=self._config.server_version,
            capabilities=self._server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            ),
        )

    @property
    def server(self) -> Server:
        return self._server

    @property
    def guardian(self) -> Guardian | None:
        return self._guardian

    async def teardown(self) -> None:
        await self._upstream.close()
        if self._db:
            await self._db.close()
        logger.info("proxy_shutdown")
