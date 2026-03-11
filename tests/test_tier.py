"""Tests for the feature tier system."""
from __future__ import annotations

import pytest

from janus.tier import Tier, TierError, current_tier
from tests.conftest import TEST_PRO_KEY


@pytest.fixture(autouse=True)
def _reset_tier():
    current_tier.reset()
    yield
    current_tier.reset()


def test_default_tier_is_free() -> None:
    assert current_tier.tier == Tier.FREE
    assert not current_tier.is_pro


def test_activate_pro() -> None:
    result = current_tier.activate(TEST_PRO_KEY)
    assert result is True
    assert current_tier.is_pro
    assert current_tier.tier == Tier.PRO


def test_activate_invalid_key() -> None:
    result = current_tier.activate("bad-key")
    assert result is False
    assert not current_tier.is_pro


def test_free_features_always_available() -> None:
    assert current_tier.check("rule_based_risk") is True
    assert current_tier.check("permission_checks") is True
    assert current_tier.check("prompt_injection_regex") is True
    assert current_tier.check("circuit_breaker") is True
    assert current_tier.check("mcp_proxy") is True


def test_pro_features_blocked_on_free() -> None:
    assert current_tier.check("llm_classifier") is False
    assert current_tier.check("drift_detection") is False
    assert current_tier.check("taint_tracking") is False
    assert current_tier.check("predictive_risk") is False
    assert current_tier.check("crowd_threat_intel") is False


def test_pro_features_available_on_pro() -> None:
    current_tier.activate(TEST_PRO_KEY)
    assert current_tier.check("llm_classifier") is True
    assert current_tier.check("drift_detection") is True
    assert current_tier.check("taint_tracking") is True
    assert current_tier.check("predictive_risk") is True


def test_require_raises_on_free() -> None:
    with pytest.raises(TierError, match="requires Janus Pro"):
        current_tier.require("llm_classifier")


def test_require_passes_on_pro() -> None:
    current_tier.activate(TEST_PRO_KEY)
    current_tier.require("llm_classifier")  # should not raise


def test_reset() -> None:
    current_tier.activate(TEST_PRO_KEY)
    assert current_tier.is_pro
    current_tier.reset()
    assert not current_tier.is_pro
