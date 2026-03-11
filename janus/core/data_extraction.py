"""Data extraction / volume tracking for the security pipeline.

Detects bulk data access, rapid sequential reads, and unbounded queries
that may indicate data exfiltration attempts.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import structlog

from janus.core.decision import (
    CheckResult,
    PipelineContext,
    ToolCallRequest,
)
from janus.risk import thresholds

logger = structlog.get_logger()

# Tools that count as "data read" operations
_READ_TOOLS: frozenset[str] = frozenset({
    "read_file", "list_files", "search_web", "database_query",
})

# Regex to detect unbounded SELECT queries
_SELECT_STAR_NO_LIMIT = re.compile(
    r"SELECT\s+\*\s+FROM\s+\w+(?:\s+WHERE\s+[^;]+)?(?:\s*;?\s*)$",
    re.IGNORECASE,
)
_HAS_LIMIT = re.compile(r"\bLIMIT\b", re.IGNORECASE)


@dataclass
class SessionDataMetrics:
    """Cumulative data metrics for a single session."""

    total_bytes_accessed: int = 0
    read_count: int = 0
    query_count: int = 0
    access_log: list[tuple[str, int, float]] = field(default_factory=list)


class DataVolumeTracker:
    """Tracks cumulative data volume per session."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionDataMetrics] = {}

    def record_access(
        self, session_id: str, tool_name: str, data_size: int
    ) -> SessionDataMetrics:
        metrics = self._sessions.setdefault(session_id, SessionDataMetrics())
        metrics.total_bytes_accessed += data_size
        metrics.access_log.append((tool_name, data_size, time.monotonic()))

        if tool_name in _READ_TOOLS:
            metrics.read_count += 1
        if tool_name == "database_query":
            metrics.query_count += 1

        return metrics

    def get_metrics(self, session_id: str) -> SessionDataMetrics:
        return self._sessions.get(session_id, SessionDataMetrics())

    def get_recent_access_count(
        self, session_id: str, window_seconds: float = 120.0
    ) -> int:
        metrics = self._sessions.get(session_id)
        if not metrics:
            return 0
        cutoff = time.monotonic() - window_seconds
        return sum(1 for _, _, ts in metrics.access_log if ts >= cutoff)


def _estimate_data_size(request: ToolCallRequest) -> int:
    """Estimate data size from tool_input hints or fallback to serialized size."""
    ti = request.tool_input

    # Check explicit size hints
    if "size" in ti and isinstance(ti["size"], (int, float)):
        return int(ti["size"])
    if "rows" in ti and isinstance(ti["rows"], (int, float)):
        return int(ti["rows"]) * 200  # ~200 bytes per row estimate

    # Fallback: serialized size of tool_input
    try:
        return len(json.dumps(ti))
    except (TypeError, ValueError):
        return 100  # safe fallback


def _is_unbounded_query(request: ToolCallRequest) -> bool:
    """Check if a database query is a SELECT * without LIMIT."""
    if request.tool_name != "database_query":
        return False
    sql = request.tool_input.get("sql", "")
    if not isinstance(sql, str):
        return False
    if _SELECT_STAR_NO_LIMIT.search(sql) and not _HAS_LIMIT.search(sql):
        return True
    return False


def _has_large_result(request: ToolCallRequest) -> bool:
    """Check if query result exceeds 10K rows."""
    rows = request.tool_input.get("rows")
    if isinstance(rows, (int, float)) and rows >= 10_000:
        return True
    return False


class DataVolumeCheck:
    """Pipeline check that monitors cumulative data volume per session.

    Priority 22: after permission checks, before deterministic risk.
    """

    name: str = "data_volume"
    priority: int = 22

    # Thresholds
    VOLUME_WARNING_BYTES = 500 * 1024       # 500 KB
    VOLUME_BLOCK_BYTES = 2 * 1024 * 1024    # 2 MB
    BULK_ACCESS_THRESHOLD = 10              # reads in window
    BULK_ACCESS_WINDOW = 120.0              # seconds

    def __init__(self, tracker: DataVolumeTracker) -> None:
        self._tracker = tracker
        # Track per-session whether bulk access has already been penalized.
        # The penalty fires ONCE per session on the first action tool after
        # the threshold is crossed, not on every subsequent action tool.
        self._bulk_flagged_sessions: set[str] = set()

    async def evaluate(
        self, request: ToolCallRequest, context: PipelineContext
    ) -> CheckResult:
        data_size = _estimate_data_size(request)
        metrics = self._tracker.record_access(
            request.session_id, request.tool_name, data_size
        )

        signals: list[str] = []
        risk = 0.0

        # Read-only tools (read_file, search_web, list_files, send_message)
        # always record access for tracking, but ONLY materialise risk when
        # the current tool is an action tool.  An agent reading 20 files
        # during a code review is normal — risk should appear when the agent
        # tries to SEND or WRITE that data externally.
        is_action_tool = request.tool_name in thresholds.KEYWORD_SENSITIVE_TOOLS

        # 1. Cumulative volume checks
        if metrics.total_bytes_accessed >= self.VOLUME_BLOCK_BYTES:
            signals.append(
                f"Extreme data volume: {metrics.total_bytes_accessed / 1024:.0f} KB accessed"
            )
            if is_action_tool:
                risk += 30.0
        elif metrics.total_bytes_accessed >= self.VOLUME_WARNING_BYTES:
            signals.append(
                f"High data volume: {metrics.total_bytes_accessed / 1024:.0f} KB accessed"
            )
            if is_action_tool:
                risk += 10.0

        # 2. Rapid sequential access — penalise ONCE per session
        recent_count = self._tracker.get_recent_access_count(
            request.session_id, self.BULK_ACCESS_WINDOW
        )
        if recent_count >= self.BULK_ACCESS_THRESHOLD:
            signals.append(
                f"Rapid sequential access: {recent_count} reads in {self.BULK_ACCESS_WINDOW:.0f}s"
            )
            if is_action_tool and request.session_id not in self._bulk_flagged_sessions:
                risk += 15.0
                self._bulk_flagged_sessions.add(request.session_id)

        # 3. Unbounded query detection (always action tool: database_query)
        if _is_unbounded_query(request):
            signals.append("Unbounded SELECT * query without LIMIT")
            risk += 10.0

        # 4. Very large query result (always action tool: database_query)
        if _has_large_result(request):
            signals.append(
                f"Very large query result: {request.tool_input.get('rows')} rows"
            )
            risk += 25.0

        reason = "; ".join(signals) if signals else "Data volume within normal limits."

        return CheckResult(
            check_name=self.name,
            passed=len(signals) == 0,
            risk_contribution=risk,
            reason=reason,
            metadata={
                "total_bytes": metrics.total_bytes_accessed,
                "read_count": metrics.read_count,
                "recent_access_count": recent_count if signals else 0,
                "signals": signals,
            },
        )
