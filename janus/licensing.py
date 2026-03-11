"""HMAC-signed license key validation for Janus Pro."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
import time

# Lazy-loaded verification keys — no hardcoded fallback in production.
_verification_key: bytes | None = None
_legacy_verification_key: bytes | None = None


def _get_verification_key() -> bytes:
    """Resolve the license signing/verification key lazily.

    - If JANUS_LICENSE_SECRET is set, use it.
    - Under pytest, fall back to a test-only key.
    - Otherwise raise RuntimeError (forces production deployments to configure it).
    """
    global _verification_key
    if _verification_key is not None:
        return _verification_key

    secret = os.environ.get("JANUS_LICENSE_SECRET")
    if secret:
        _verification_key = secret.encode()
        return _verification_key

    if "pytest" in sys.modules:
        _verification_key = b"janus-test-key"
        return _verification_key

    raise RuntimeError(
        "JANUS_LICENSE_SECRET is not set. "
        "Generate one with: openssl rand -hex 32"
    )


def _get_legacy_verification_key() -> bytes:
    """Resolve the legacy (sentinel-era) verification key lazily."""
    global _legacy_verification_key
    if _legacy_verification_key is not None:
        return _legacy_verification_key

    secret = os.environ.get("JANUS_LEGACY_LICENSE_SECRET")
    if secret:
        _legacy_verification_key = secret.encode()
        return _legacy_verification_key

    if "pytest" in sys.modules:
        _legacy_verification_key = b"janus-test-key"
        return _legacy_verification_key

    raise RuntimeError(
        "JANUS_LEGACY_LICENSE_SECRET is not set. "
        "Generate one with: openssl rand -hex 32"
    )


def _reset_verification_key() -> None:
    """Reset cached keys so they are re-read from env on next access.

    Useful for tests and dev-mode bootstrap.
    """
    global _verification_key, _legacy_verification_key
    _verification_key = None
    _legacy_verification_key = None


def validate_license(key: str) -> tuple[bool, str]:
    """Validate an HMAC-signed license key.

    Accepts both sk-janus- and legacy sk-sentinel- prefixes.
    Returns (is_valid, tier).
    """
    if key.startswith("sk-janus-"):
        prefix = "sk-janus-"
        verification_key = _get_verification_key()
    elif key.startswith("sk-sentinel-"):
        prefix = "sk-sentinel-"
        verification_key = _get_legacy_verification_key()
    else:
        return False, "free"

    remainder = key[len(prefix):]
    parts = remainder.rsplit("-", 1)
    if len(parts) != 2:
        return False, "free"

    payload_b64, sig_hex = parts

    expected = hmac.new(
        verification_key, payload_b64.encode(), hashlib.sha256
    ).hexdigest()[:16]

    if not hmac.compare_digest(sig_hex, expected):
        return False, "free"

    try:
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64 + "==")
        )
    except (ValueError, json.JSONDecodeError):
        return False, "free"

    if payload.get("exp", 0) and time.time() > payload["exp"]:
        return False, "free"

    return True, payload.get("tier", "pro")


async def is_license_revoked(key: str, db: object) -> bool:
    """Check if a license has been revoked/expired in the database.

    Returns True if the license exists in DB with a non-active status.
    Returns False (not revoked) if the key isn't in the DB at all
    (e.g. dev-mode keys, test keys).
    """
    if db is None:
        return False
    row = await db.fetchone(  # type: ignore[union-attr]
        "SELECT status FROM licenses WHERE license_key = ?",
        (key,),
    )
    if row is None:
        return False  # Not a DB-issued key (dev/test) — allow it
    return row[0] != "active"


def generate_license(
    tier: str = "pro",
    customer_id: str = "",
    expiry_days: int = 365,
    signing_key: bytes | None = None,
) -> str:
    """Generate a signed license key. For internal use."""
    if signing_key is None:
        signing_key = _get_verification_key()
    payload = {
        "tier": tier,
        "cid": customer_id,
        "exp": int(time.time()) + expiry_days * 86400,
    }
    payload_b64 = (
        base64.urlsafe_b64encode(json.dumps(payload).encode())
        .rstrip(b"=")
        .decode()
    )
    sig = hmac.new(
        signing_key, payload_b64.encode(), hashlib.sha256
    ).hexdigest()[:16]
    return f"sk-janus-{payload_b64}-{sig}"
