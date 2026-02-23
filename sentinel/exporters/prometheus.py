"""Prometheus metrics exporter for Sentinel.

Exposes counters, gauges, and histograms for monitoring.
Uses internal tracking if prometheus_client is not installed.
"""
from __future__ import annotations

from collections import defaultdict

from sentinel.core.decision import SecurityVerdict

try:
    from prometheus_client import Counter, Gauge, Histogram

    HAS_PROM = True
except ImportError:
    HAS_PROM = False


class PrometheusExporter:
    """Records Sentinel metrics for Prometheus scraping."""

    def __init__(self) -> None:
        # Internal tracking (always available)
        self._verdict_counts: dict[str, int] = defaultdict(int)
        self._risk_scores: dict[str, float] = {}
        self._tool_counts: dict[str, int] = defaultdict(int)

        # Prometheus native metrics (if available)
        if HAS_PROM:
            self._prom_verdicts = Counter(
                "sentinel_verdicts_total",
                "Total verdicts by type",
                ["verdict"],
            )
            self._prom_risk = Gauge(
                "sentinel_session_risk_score",
                "Current session risk score",
                ["session_id"],
            )
            self._prom_duration = Histogram(
                "sentinel_intercept_duration_seconds",
                "Time to process an interception",
            )
            self._prom_tools = Counter(
                "sentinel_tool_calls_total",
                "Total tool calls by name",
                ["tool_name"],
            )

    def record(
        self,
        verdict: SecurityVerdict,
        tool_name: str = "",
        session_id: str = "",
        duration_ms: float | None = None,
    ) -> None:
        """Record a verdict in metrics."""
        v = verdict.verdict.value
        self._verdict_counts[v] += 1
        if tool_name:
            self._tool_counts[tool_name] += 1
        if session_id:
            self._risk_scores[session_id] = verdict.risk_score

        if HAS_PROM:
            self._prom_verdicts.labels(verdict=v).inc()
            if tool_name:
                self._prom_tools.labels(tool_name=tool_name).inc()
            if session_id:
                self._prom_risk.labels(session_id=session_id).set(
                    verdict.risk_score
                )
            if duration_ms is not None:
                self._prom_duration.observe(duration_ms / 1000.0)

    def get_verdict_count(self, verdict: str) -> int:
        return self._verdict_counts.get(verdict, 0)

    def get_risk_score(self, session_id: str) -> float:
        return self._risk_scores.get(session_id, 0.0)
