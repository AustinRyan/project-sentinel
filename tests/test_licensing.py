"""Tests for HMAC-signed license validation."""
from __future__ import annotations

from janus.licensing import generate_license, validate_license
from tests.conftest import TEST_PRO_KEY


def test_valid_key():
    valid, tier = validate_license(TEST_PRO_KEY)
    assert valid is True
    assert tier == "pro"


def test_freshly_generated_key():
    key = generate_license(tier="pro", customer_id="ci")
    valid, tier = validate_license(key)
    assert valid is True
    assert tier == "pro"


def test_invalid_prefix():
    valid, tier = validate_license("bad-key-123")
    assert valid is False
    assert tier == "free"


def test_invalid_signature():
    valid, tier = validate_license("sk-janus-fakepayload-0000000000000000")
    assert valid is False
    assert tier == "free"


def test_tampered_payload():
    key = generate_license(tier="pro")
    # Tamper with the payload (change a character)
    parts = key.rsplit("-", 1)
    payload = parts[0][len("sk-janus-"):]
    tampered = payload[:-1] + ("A" if payload[-1] != "A" else "B")
    tampered_key = f"sk-janus-{tampered}-{parts[1]}"
    valid, tier = validate_license(tampered_key)
    assert valid is False
    assert tier == "free"


def test_expired_key():
    key = generate_license(tier="pro", expiry_days=-1)
    valid, tier = validate_license(key)
    assert valid is False
    assert tier == "free"


def test_missing_signature_part():
    valid, tier = validate_license("sk-janus-justpayloadnosig")
    assert valid is False
    assert tier == "free"


def test_custom_signing_key():
    custom_key = b"my-custom-key"
    key = generate_license(signing_key=custom_key)
    # Should fail with default verification key
    valid, _ = validate_license(key)
    assert valid is False
