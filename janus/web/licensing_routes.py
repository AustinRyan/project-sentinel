"""Licensing API routes — Stripe checkout, webhook, session lookup, license activation."""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from janus.licensing import generate_license, is_license_revoked, validate_license
from janus.tier import current_tier
from janus.web.auth import rate_limit_license, require_api_key

logger = structlog.get_logger()

router = APIRouter(prefix="/api")


class ActivateRequest(BaseModel):
    license_key: str


class CheckoutRequest(BaseModel):
    email: str | None = None


# ---------------------------------------------------------------------------
# License status & activation
# ---------------------------------------------------------------------------


@router.get("/license/status")
async def license_status() -> dict:
    return {
        "tier": current_tier.tier.value,
        "is_pro": current_tier.is_pro,
    }


@router.post("/license/activate", dependencies=[Depends(require_api_key), Depends(rate_limit_license)])
async def license_activate(req: ActivateRequest) -> JSONResponse:
    valid, tier = validate_license(req.license_key)
    if not valid:
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid or expired license key"},
        )

    # Check if this license was revoked (subscription cancelled)
    from janus.web.app import state

    if await is_license_revoked(req.license_key, state.db):
        return JSONResponse(
            status_code=403,
            content={"detail": "This license has been cancelled. Please purchase a new subscription."},
        )

    current_tier.activate(req.license_key)

    # Rebuild Guardian pipeline so PRO checks take effect immediately
    from janus.web.app import state

    if state.guardian is not None:
        state.guardian.rebuild_pipeline()

    logger.info("license_activated_via_api", tier=tier)
    return JSONResponse(
        status_code=200,
        content={"tier": tier, "is_pro": tier == "pro"},
    )


# ---------------------------------------------------------------------------
# Stripe checkout session creation
# ---------------------------------------------------------------------------


@router.post("/billing/checkout")
async def create_checkout(req: CheckoutRequest) -> JSONResponse:
    secret_key = os.environ.get("STRIPE_SECRET_KEY")
    price_id = os.environ.get("STRIPE_PRICE_ID")

    if not secret_key or not price_id:
        return JSONResponse(
            status_code=501,
            content={"detail": "Stripe billing not configured. Set STRIPE_SECRET_KEY and STRIPE_PRICE_ID."},
        )

    try:
        import stripe  # type: ignore[import-untyped]
    except ImportError:
        return JSONResponse(
            status_code=501,
            content={
                "detail": (
                    "stripe package not installed. "
                    "Install with: pip install 'janus-security[billing]'"
                )
            },
        )

    stripe.api_key = secret_key

    base_url = os.environ.get("JANUS_BASE_URL", "http://localhost:3000")
    session_params: dict = {
        "mode": "subscription",
        "line_items": [{"price": price_id, "quantity": 1}],
        "subscription_data": {"trial_period_days": 14},
        "success_url": f"{base_url}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}",
        "cancel_url": f"{base_url}/landing#pricing",
    }
    if req.email:
        session_params["customer_email"] = req.email

    try:
        session = stripe.checkout.Session.create(**session_params)
    except stripe.error.StripeError as e:
        logger.error("stripe_checkout_error", error=str(e))
        return JSONResponse(
            status_code=502,
            content={"detail": "Failed to create checkout session."},
        )

    return JSONResponse(status_code=200, content={"checkout_url": session.url})


# ---------------------------------------------------------------------------
# Session lookup (for success page)
# ---------------------------------------------------------------------------


@router.get("/billing/session/{session_id}")
async def get_billing_session(session_id: str) -> JSONResponse:
    from janus.web.app import state

    if state.db is None:
        return JSONResponse(status_code=503, content={"detail": "Database not available."})

    row = await state.db.fetchone(
        "SELECT license_key, tier, customer_email, created_at FROM licenses WHERE stripe_session_id = ?",
        (session_id,),
    )
    if row is None:
        return JSONResponse(status_code=404, content={"detail": "Session not found."})

    created_at = row[3] or ""
    trial_ends_at = ""
    if created_at:
        try:
            dt = datetime.fromisoformat(created_at).replace(tzinfo=UTC)
            trial_ends_at = (dt + timedelta(days=14)).isoformat()
        except (ValueError, TypeError):
            pass

    return JSONResponse(
        status_code=200,
        content={
            "license_key": row[0],
            "tier": row[1],
            "customer_email": row[2],
            "trial_ends_at": trial_ends_at,
        },
    )


# ---------------------------------------------------------------------------
# Stripe webhook
# ---------------------------------------------------------------------------


@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request) -> JSONResponse:
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    if not webhook_secret:
        return JSONResponse(
            status_code=501,
            content={"detail": "Stripe webhook not configured. Set STRIPE_WEBHOOK_SECRET."},
        )

    try:
        import stripe  # type: ignore[import-untyped]
    except ImportError:
        return JSONResponse(
            status_code=501,
            content={
                "detail": (
                    "stripe package not installed. "
                    "Install with: pip install 'janus-security[billing]'"
                )
            },
        )

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except (ValueError, stripe.error.SignatureVerificationError):
        return JSONResponse(status_code=400, content={"detail": "Invalid signature"})

    from janus.web.app import state

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_email = session.get("customer_email") or session.get("customer_details", {}).get("email") or ""
        stripe_customer_id = session.get("customer") or ""
        stripe_session_id = session.get("id") or ""

        license_key = generate_license(tier="pro", customer_id=customer_email)

        if state.db is not None:
            await state.db.execute(
                """
                INSERT INTO licenses
                    (license_key, tier, customer_email, stripe_customer_id, stripe_session_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (license_key, "pro", customer_email, stripe_customer_id, stripe_session_id),
            )
            await state.db.commit()

        # Send license email (best-effort)
        from janus.email import send_license_email

        if customer_email:
            send_license_email(to=customer_email, license_key=license_key, tier="pro")

        logger.info(
            "stripe_license_generated",
            email=customer_email,
            session_id=stripe_session_id,
        )
        return JSONResponse(
            status_code=200,
            content={"license_key": license_key, "tier": "pro"},
        )

    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        stripe_customer_id = subscription.get("customer", "")

        if state.db is not None and stripe_customer_id:
            await state.db.execute(
                "UPDATE licenses SET status = 'expired' WHERE stripe_customer_id = ? AND status = 'active'",
                (stripe_customer_id,),
            )
            await state.db.commit()
            logger.info("stripe_subscription_cancelled", customer_id=stripe_customer_id)

        return JSONResponse(status_code=200, content={"received": True, "action": "license_expired"})

    return JSONResponse(status_code=200, content={"received": True})
