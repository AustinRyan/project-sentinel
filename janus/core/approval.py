"""Human-in-the-loop approval manager for blocked/challenged tool calls."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import structlog

from janus.storage.database import DatabaseManager
from janus.web.events import EventBroadcaster, SecurityEvent

# Checks that represent hard policy violations — no human review needed.
# If any of these forced the block, it's a clear denial.
HARD_BLOCK_CHECKS = frozenset({
    "permission_scope",
    "identity_check",
    "prompt_injection",
})

# Verdicts that always warrant human review
REVIEW_VERDICTS = frozenset({"challenge", "pause"})

logger = structlog.get_logger()


def needs_human_review(verdict: str, check_results: list[dict[str, Any]]) -> bool:
    """Determine if a non-ALLOW verdict warrants human review.

    Returns True for judgment calls (high risk, drift, suspicious patterns).
    Returns False for hard policy violations (permission denied, injection, identity).
    """
    # SANDBOX means sandboxed execution, not human review
    if verdict == "sandbox":
        return False

    # Check if any hard-policy check forced the verdict — these are clear
    # denials (permission denied, injection detected, identity failure)
    # regardless of whether the final verdict is block or challenge.
    for cr in check_results:
        if cr.get("check_name") in HARD_BLOCK_CHECKS and not cr.get("passed", True):
            return False

    # CHALLENGE and PAUSE without a hard-policy cause → human review
    if verdict in REVIEW_VERDICTS:
        return True

    # BLOCK from risk accumulation / LLM classifier / drift / threat intel
    # — these are judgment calls that a human might override
    return True


@dataclass
class ApprovalRequest:
    """A pending approval request for a blocked tool call."""

    id: str
    session_id: str
    agent_id: str
    tool_name: str
    tool_input: dict[str, Any]
    original_goal: str
    verdict: str
    risk_score: float
    risk_delta: float
    reasons: list[str]
    check_results: list[dict[str, Any]]
    trace_id: str
    status: str  # pending, approved, rejected
    decided_by: str | None
    decided_at: str | None
    decision_reason: str
    tool_result: dict[str, Any] | None
    created_at: str
    expires_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ApprovalManager:
    """Manages human-in-the-loop approval requests for non-ALLOW verdicts."""

    def __init__(
        self,
        db: DatabaseManager,
        broadcaster: EventBroadcaster,
        tool_executor: Any | None = None,
        guardian: Any | None = None,
        exporter_coordinator: Any | None = None,
    ) -> None:
        self._db = db
        self._broadcaster = broadcaster
        self._tool_executor = tool_executor
        self._guardian = guardian
        self._exporter_coordinator = exporter_coordinator

    async def create(
        self,
        *,
        session_id: str,
        agent_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        original_goal: str,
        verdict: str,
        risk_score: float,
        risk_delta: float,
        reasons: list[str],
        check_results: list[dict[str, Any]],
        trace_id: str = "",
    ) -> ApprovalRequest:
        """Create a new approval request for a blocked tool call."""
        approval_id = f"apr-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC).isoformat()

        await self._db.execute(
            """INSERT INTO approval_requests
            (id, session_id, agent_id, tool_name, tool_input_json, original_goal,
             verdict, risk_score, risk_delta, reasons_json, check_results_json,
             trace_id, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (
                approval_id,
                session_id,
                agent_id,
                tool_name,
                json.dumps(tool_input, default=str),
                original_goal,
                verdict,
                risk_score,
                risk_delta,
                json.dumps(reasons),
                json.dumps(check_results, default=str),
                trace_id,
                now,
            ),
        )
        await self._db.commit()

        request = ApprovalRequest(
            id=approval_id,
            session_id=session_id,
            agent_id=agent_id,
            tool_name=tool_name,
            tool_input=tool_input,
            original_goal=original_goal,
            verdict=verdict,
            risk_score=risk_score,
            risk_delta=risk_delta,
            reasons=reasons,
            check_results=check_results,
            trace_id=trace_id,
            status="pending",
            decided_by=None,
            decided_at=None,
            decision_reason="",
            tool_result=None,
            created_at=now,
            expires_at=None,
        )

        # Broadcast to monitor dashboards
        await self._broadcaster.publish(SecurityEvent(
            event_type="approval_created",
            session_id=session_id,
            data=request.to_dict(),
        ))

        # Fire notifiers
        if self._exporter_coordinator is not None:
            try:
                from janus.core.decision import SecurityVerdict, Verdict
                sv = SecurityVerdict(
                    verdict=Verdict(verdict),
                    risk_score=risk_score,
                    risk_delta=risk_delta,
                    reasons=[f"[APPROVAL NEEDED] {r}" for r in reasons],
                )
                await self._exporter_coordinator.export(
                    sv,
                    tool_name=tool_name,
                    agent_id=agent_id,
                    session_id=session_id,
                )
            except Exception:
                logger.exception("approval_notify_error")

        return request

    async def get_pending(
        self,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[ApprovalRequest]:
        """Get pending approval requests, optionally filtered by session."""
        if session_id:
            rows = await self._db.fetchall(
                "SELECT * FROM approval_requests WHERE status = 'pending' AND session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            )
        else:
            rows = await self._db.fetchall(
                "SELECT * FROM approval_requests WHERE status = 'pending' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [self._row_to_request(row) for row in rows]

    async def get_all(
        self,
        status: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[ApprovalRequest]:
        """Get approval requests with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await self._db.fetchall(
            f"SELECT * FROM approval_requests{where} ORDER BY created_at DESC LIMIT ?",
            tuple(params),
        )
        return [self._row_to_request(row) for row in rows]

    async def get_by_id(self, approval_id: str) -> ApprovalRequest | None:
        """Get a single approval request by ID."""
        row = await self._db.fetchone(
            "SELECT * FROM approval_requests WHERE id = ?",
            (approval_id,),
        )
        if row is None:
            return None
        return self._row_to_request(row)

    async def approve(
        self,
        approval_id: str,
        decided_by: str = "human",
        reason: str = "",
    ) -> ApprovalRequest | None:
        """Approve a pending request: execute the tool and return the result."""
        request = await self.get_by_id(approval_id)
        if request is None or request.status != "pending":
            return None

        now = datetime.now(UTC).isoformat()
        tool_result: dict[str, Any] | None = None

        # Execute the tool
        if self._tool_executor is not None:
            try:
                tool_result = await self._tool_executor.execute(
                    request.tool_name, request.tool_input
                )
            except Exception as exc:
                logger.exception("approval_execute_error")
                tool_result = {"error": str(exc)}

        # Scan output for taints if guardian is available
        if tool_result and self._guardian is not None:
            try:
                if hasattr(self._guardian, "taint_tracker"):
                    self._guardian.taint_tracker.scan_output(
                        request.session_id,
                        request.tool_name,
                        tool_result,
                        step=0,
                    )
            except Exception:
                logger.exception("approval_taint_scan_error")

        await self._db.execute(
            """UPDATE approval_requests
            SET status = 'approved', decided_by = ?, decided_at = ?,
                decision_reason = ?, tool_result_json = ?
            WHERE id = ?""",
            (decided_by, now, reason, json.dumps(tool_result, default=str) if tool_result else None, approval_id),
        )
        await self._db.commit()

        request.status = "approved"
        request.decided_by = decided_by
        request.decided_at = now
        request.decision_reason = reason
        request.tool_result = tool_result

        # Broadcast resolution
        await self._broadcaster.publish(SecurityEvent(
            event_type="approval_resolved",
            session_id=request.session_id,
            data={
                **request.to_dict(),
                "resolution": "approved",
            },
        ))

        return request

    async def reject(
        self,
        approval_id: str,
        decided_by: str = "human",
        reason: str = "",
    ) -> ApprovalRequest | None:
        """Reject a pending request."""
        request = await self.get_by_id(approval_id)
        if request is None or request.status != "pending":
            return None

        now = datetime.now(UTC).isoformat()

        await self._db.execute(
            """UPDATE approval_requests
            SET status = 'rejected', decided_by = ?, decided_at = ?,
                decision_reason = ?
            WHERE id = ?""",
            (decided_by, now, reason, approval_id),
        )
        await self._db.commit()

        request.status = "rejected"
        request.decided_by = decided_by
        request.decided_at = now
        request.decision_reason = reason

        # Broadcast resolution
        await self._broadcaster.publish(SecurityEvent(
            event_type="approval_resolved",
            session_id=request.session_id,
            data={
                **request.to_dict(),
                "resolution": "rejected",
            },
        ))

        return request

    async def get_stats(self) -> dict[str, int]:
        """Get counts by status."""
        rows = await self._db.fetchall(
            "SELECT status, COUNT(*) as count FROM approval_requests GROUP BY status"
        )
        stats = {"pending": 0, "approved": 0, "rejected": 0}
        for row in rows:
            stats[row["status"]] = row["count"]
        stats["total"] = sum(stats.values())
        return stats

    def _row_to_request(self, row: Any) -> ApprovalRequest:
        """Convert a database row to an ApprovalRequest."""
        return ApprovalRequest(
            id=row["id"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            tool_name=row["tool_name"],
            tool_input=json.loads(row["tool_input_json"]),
            original_goal=row["original_goal"],
            verdict=row["verdict"],
            risk_score=row["risk_score"],
            risk_delta=row["risk_delta"],
            reasons=json.loads(row["reasons_json"]),
            check_results=json.loads(row["check_results_json"]),
            trace_id=row["trace_id"],
            status=row["status"],
            decided_by=row["decided_by"],
            decided_at=row["decided_at"],
            decision_reason=row["decision_reason"],
            tool_result=json.loads(row["tool_result_json"]) if row["tool_result_json"] else None,
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )
