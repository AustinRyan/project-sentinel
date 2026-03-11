from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import time_machine

from janus.risk.engine import RiskEngine
from janus.risk.patterns import PatternDetector
from janus.risk.scoring import RiskScorer
from janus.risk.thresholds import (
    DEFAULT_TOOL_BASE_RISK,
    KEYWORD_AMPLIFIER_CAP,
    LOCK_THRESHOLD,
    MAX_RISK_SCORE,
    MIN_RISK_SCORE,
    TOOL_BASE_RISK,
    VELOCITY_PENALTY_CAP,
    VELOCITY_PENALTY_PER_CALL,
    VELOCITY_THRESHOLD_CALLS,
)
from janus.storage.session_store import InMemorySessionStore, RiskEvent

# ── fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def scorer() -> RiskScorer:
    return RiskScorer()


@pytest.fixture
def detector() -> PatternDetector:
    return PatternDetector()


@pytest.fixture
def engine(session_store: InMemorySessionStore) -> RiskEngine:
    return RiskEngine(session_store=session_store)


# ── RiskScorer tests ────────────────────────────────────────────────


class TestBaseRisk:
    def test_base_risk_lookup(self, scorer: RiskScorer) -> None:
        """Known tools return the correct base risk from TOOL_BASE_RISK."""
        for tool_name, expected in TOOL_BASE_RISK.items():
            assert scorer._get_base_risk(tool_name) == expected

    def test_default_risk_for_unknown_tool(self, scorer: RiskScorer) -> None:
        """An unknown tool name returns DEFAULT_TOOL_BASE_RISK."""
        assert scorer._get_base_risk("totally_unknown_tool") == DEFAULT_TOOL_BASE_RISK


class TestKeywordScanning:
    def test_keyword_scanning_finds_matches(self, scorer: RiskScorer) -> None:
        """Dangerous patterns in sensitive tool inputs are detected and scored."""
        tool_input = {"code": "os.system('rm -rf /')"}
        result = scorer._scan_keywords("execute_code", tool_input)
        assert result > 0.0

    def test_keyword_scanning_skips_benign_tools(self, scorer: RiskScorer) -> None:
        """Benign tools are exempt from keyword scanning."""
        tool_input = {"query": "os.system rm -rf /etc/shadow"}
        assert scorer._scan_keywords("search_web", tool_input) == 0.0
        assert scorer._scan_keywords("read_file", tool_input) == 0.0
        assert scorer._scan_keywords("send_message", tool_input) == 0.0

    def test_keyword_scanning_cap(self, scorer: RiskScorer) -> None:
        """Multiple keyword matches do not exceed KEYWORD_AMPLIFIER_CAP."""
        tool_input = {
            "a": "os.system eval( exec(",
            "b": "rm -rf /etc/shadow /etc/passwd",
            "c": "reverse_shell netcat -e exfil",
            "d": "subprocess.call chmod 777 drop table truncate table",
        }
        result = scorer._scan_keywords("execute_code", tool_input)
        assert result == KEYWORD_AMPLIFIER_CAP


class TestVelocityPenalty:
    def test_velocity_penalty_below_threshold(self, scorer: RiskScorer) -> None:
        """No penalty when event count is at or below the threshold."""
        now = datetime.now(UTC)
        events = [
            RiskEvent(risk_delta=1.0, new_score=1.0, tool_name="read_file", reason="test", timestamp=now)
            for _ in range(VELOCITY_THRESHOLD_CALLS)
        ]
        assert scorer._velocity_penalty(events) == 0.0

    def test_velocity_penalty_above_threshold(self, scorer: RiskScorer) -> None:
        """Penalty applied when event count exceeds threshold."""
        now = datetime.now(UTC)
        extra = 3
        events = [
            RiskEvent(risk_delta=1.0, new_score=1.0, tool_name="read_file", reason="test", timestamp=now)
            for _ in range(VELOCITY_THRESHOLD_CALLS + extra)
        ]
        expected = min(VELOCITY_PENALTY_CAP, extra * VELOCITY_PENALTY_PER_CALL)
        assert scorer._velocity_penalty(events) == expected


# ── PatternDetector tests ───────────────────────────────────────────


class TestSleeperPattern:
    def test_sleeper_pattern_step1(self, detector: PatternDetector) -> None:
        """First step alone yields low risk contribution."""
        result = detector.match(
            tool_name="read_file",
            tool_input={"path": "api_docs.md"},
            session_history=[],
        )
        assert result.matched_steps == 1
        assert result.matched is False
        assert result.risk_contribution > 0
        assert result.risk_contribution < 20  # low

    def test_sleeper_pattern_step2(self, detector: PatternDetector) -> None:
        """Two steps yield medium risk with amplification."""
        history: list[tuple[str, dict]] = [
            ("read_file", {"path": "api_docs.md"}),
        ]
        result = detector.match(
            tool_name="search_web",
            tool_input={"query": "auth token endpoint"},
            session_history=history,
        )
        assert result.matched_steps == 2
        assert result.matched is False
        assert result.risk_contribution > result.matched_steps  # amplified

    def test_sleeper_pattern_full_match(self, detector: PatternDetector) -> None:
        """All three steps matched signals high risk that should trigger block."""
        history: list[tuple[str, dict]] = [
            ("read_file", {"path": "api_docs.md"}),
            ("search_web", {"query": "auth token endpoint"}),
        ]
        result = detector.match(
            tool_name="execute_code",
            tool_input={"code": "curl -X POST login_test"},
            session_history=history,
        )
        assert result.matched is True
        assert result.pattern_name == "sleeper_reconnaissance"
        assert result.matched_steps == 3
        assert result.risk_contribution >= 60  # high enough to contribute to lock


# ── RiskEngine tests ────────────────────────────────────────────────


class TestRiskEngine:
    def test_risk_engine_score_accumulation(
        self, engine: RiskEngine, session_store: InMemorySessionStore
    ) -> None:
        """Calling update_score repeatedly accumulates risk."""
        sid = "sess-accum"
        engine.update_score(sid, 30.0)
        engine.update_score(sid, 25.0)
        assert session_store.get_risk_score(sid) == 55.0

    def test_risk_engine_is_locked_at_80(
        self, engine: RiskEngine, session_store: InMemorySessionStore
    ) -> None:
        """Session is locked once score reaches LOCK_THRESHOLD (80)."""
        sid = "sess-lock"
        engine.update_score(sid, LOCK_THRESHOLD)
        assert engine.is_locked(sid) is True

    def test_risk_engine_decay_after_idle(
        self, session_store: InMemorySessionStore
    ) -> None:
        """Score decays after the idle threshold is exceeded."""
        sid = "sess-decay"
        engine = RiskEngine(session_store=session_store)

        base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        with time_machine.travel(base_time, tick=False):
            engine.update_score(sid, 50.0)
            event = RiskEvent(
                risk_delta=50.0,
                new_score=50.0,
                tool_name="write_file",
                reason="test",
                timestamp=datetime.now(UTC),
            )
            engine.add_event(sid, event)
            assert engine.get_score(sid) == 50.0

        # Advance 10 minutes into the future (5 min idle threshold + 5 min decay)
        future_time = base_time + timedelta(minutes=10)
        with time_machine.travel(future_time, tick=False):
            score = engine.get_score(sid)
            # decay = 2.0 * (10 - 5) = 10.0  =>  50 - 10 = 40
            assert score == 40.0

    def test_score_clamped_to_bounds(
        self, engine: RiskEngine, session_store: InMemorySessionStore
    ) -> None:
        """Score never goes below 0 or above 100."""
        sid = "sess-clamp"

        # Try to go above MAX
        new = engine.update_score(sid, 200.0)
        assert new == MAX_RISK_SCORE

        # Try to go below MIN
        new = engine.update_score(sid, -300.0)
        assert new == MIN_RISK_SCORE
