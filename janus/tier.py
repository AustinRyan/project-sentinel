"""Feature tier gating for Sentinel open core."""
from __future__ import annotations

from enum import StrEnum
from typing import Any

import structlog

logger = structlog.get_logger()


class Tier(StrEnum):
    FREE = "free"
    PRO = "pro"


# Features gated behind PRO tier
PRO_FEATURES: frozenset[str] = frozenset({
    "llm_classifier",
    "drift_detection",
    "taint_tracking",
    "predictive_risk",
    "crowd_threat_intel",
    "dashboard",
    "webhooks",
    "compliance_reports",
})

# Features available in FREE tier
FREE_FEATURES: frozenset[str] = frozenset({
    "rule_based_risk",
    "permission_checks",
    "identity_management",
    "prompt_injection_regex",
    "circuit_breaker",
    "basic_threat_patterns",
    "proof_chain",
    "mcp_proxy",
})


class _TierState:
    """Singleton managing the current tier and license."""

    def __init__(self) -> None:
        self._tier: Tier = Tier.FREE
        self._license_key: str = ""

    @property
    def tier(self) -> Tier:
        return self._tier

    @property
    def is_pro(self) -> bool:
        return self._tier == Tier.PRO

    def activate(self, license_key: str) -> bool:
        """Activate PRO tier with a valid HMAC-signed license key."""
        from janus.licensing import validate_license

        valid, tier = validate_license(license_key)
        if valid:
            self._tier = Tier(tier) if tier in ("free", "pro") else Tier.PRO
            self._license_key = license_key
            logger.info("tier_activated", tier=self._tier.value)
            return True
        logger.warning("tier_activation_failed", reason="invalid or expired key")
        return False

    def check(self, feature: str) -> bool:
        """Check if a feature is available in the current tier."""
        if feature in FREE_FEATURES:
            return True
        if feature in PRO_FEATURES:
            return self._tier == Tier.PRO
        return True

    def require(self, feature: str) -> None:
        """Raise if a PRO feature is used on FREE tier."""
        if not self.check(feature):
            raise TierError(
                f"'{feature}' requires Janus Pro. "
                "Upgrade at https://janus-security.dev/pricing"
            )

    def reset(self) -> None:
        """Reset to FREE tier. Used in tests."""
        self._tier = Tier.FREE
        self._license_key = ""


class TierError(Exception):
    """Raised when a PRO feature is accessed on the FREE tier."""


# Module-level singleton
current_tier = _TierState()
