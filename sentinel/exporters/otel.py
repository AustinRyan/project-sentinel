"""OpenTelemetry exporter — emits spans for each Guardian interception.

Requires: opentelemetry-api, opentelemetry-sdk (optional dependency).
Falls back to no-op if not installed.
"""
from __future__ import annotations

from typing import Any

from sentinel.core.decision import SecurityVerdict

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


class OtelExporter:
    """Records SecurityVerdicts as OpenTelemetry spans."""

    def __init__(self, service_name: str = "sentinel") -> None:
        self._service_name = service_name
        if HAS_OTEL:
            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)
            self._tracer = provider.get_tracer("sentinel.guardian")
        else:
            self._tracer = None

    def build_attributes(
        self,
        verdict: SecurityVerdict,
        tool_name: str = "",
        agent_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        return {
            "sentinel.verdict": verdict.verdict.value,
            "sentinel.risk_score": verdict.risk_score,
            "sentinel.risk_delta": verdict.risk_delta,
            "sentinel.tool_name": tool_name,
            "sentinel.agent_id": agent_id,
            "sentinel.session_id": session_id,
            "sentinel.drift_score": verdict.drift_score,
            "sentinel.trace_id": verdict.trace_id,
        }

    def record(
        self,
        verdict: SecurityVerdict,
        tool_name: str = "",
        agent_id: str = "",
        session_id: str = "",
    ) -> None:
        """Record a verdict as an OTEL span."""
        attrs = self.build_attributes(verdict, tool_name, agent_id, session_id)

        if self._tracer is not None:
            with self._tracer.start_as_current_span(
                "guardian.intercept",
                attributes=attrs,
            ):
                pass  # Span recorded with attributes
