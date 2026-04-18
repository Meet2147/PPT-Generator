import base64
import hashlib
import hmac
import os

import requests

from app.config import settings
from app.models import BillingCheckoutRequest


PLAN_MATRIX = {
    ("student", "monthly"): {
        "plan_id_env": "RAZORPAY_PLAN_STUDENT_MONTHLY",
        "plan_name": "Student Monthly",
        "description": "Student plan billed monthly",
        "amount": 80000,
    },
    ("student", "annual"): {
        "plan_id_env": "RAZORPAY_PLAN_STUDENT_ANNUAL",
        "plan_name": "Student Annual",
        "description": "Student plan billed annually",
        "amount": 790000,
    },
    ("corporate", "monthly"): {
        "plan_id_env": "RAZORPAY_PLAN_CORPORATE_MONTHLY",
        "plan_name": "Corporate Monthly",
        "description": "Corporate plan billed monthly",
        "amount": 240000,
    },
    ("corporate", "annual"): {
        "plan_id_env": "RAZORPAY_PLAN_CORPORATE_ANNUAL",
        "plan_name": "Corporate Annual",
        "description": "Corporate plan billed annually",
        "amount": 2280000,
    },
    ("executive", "monthly"): {
        "plan_id_env": "RAZORPAY_PLAN_EXECUTIVE_MONTHLY",
        "plan_name": "Executive Monthly",
        "description": "Executive plan billed monthly",
        "amount": 490000,
    },
    ("executive", "annual"): {
        "plan_id_env": "RAZORPAY_PLAN_EXECUTIVE_ANNUAL",
        "plan_name": "Executive Annual",
        "description": "Executive plan billed annually",
        "amount": 4680000,
    },
}


def create_checkout_subscription(payload: BillingCheckoutRequest) -> dict:
    _require_razorpay_config()
    meta = _plan_meta(payload.tier, payload.billing_cycle)
    response = requests.post(
        "https://api.razorpay.com/v1/subscriptions",
        headers={
            "Authorization": f"Basic {_basic_auth()}",
            "Content-Type": "application/json",
        },
        json={
            "plan_id": meta["plan_id"],
            "total_count": 1200,
            "quantity": payload.quantity,
            "customer_notify": 1,
            "notes": {
                "tier": payload.tier,
                "billing_cycle": payload.billing_cycle,
                "product": settings.app_name,
                "email": payload.email or "",
                "name": payload.name or "",
            },
        },
        timeout=45,
    )
    response.raise_for_status()
    subscription = response.json()
    return {
        "status": "created",
        "provider": "razorpay",
        "key_id": settings.razorpay_key_id,
        "subscription_id": subscription["id"],
        "plan_id": meta["plan_id"],
        "amount": meta["amount"],
        "currency": "INR",
        "plan_name": meta["plan_name"],
        "description": meta["description"],
        "customer": {
            "name": payload.name or "",
            "email": payload.email or "",
        },
    }


def verify_webhook_signature(raw_body: bytes, signature: str | None) -> bool:
    if not settings.razorpay_webhook_secret or not signature:
        return False
    digest = hmac.new(
        settings.razorpay_webhook_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(digest, signature)


def persist_webhook_event(raw_body: bytes) -> str:
    log_dir = settings.cache_dir / "billing"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "razorpay_webhooks.jsonl"
    with log_path.open("ab") as file:
        file.write(raw_body + b"\n")
    return str(log_path)


def _require_razorpay_config() -> None:
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise RuntimeError("Missing Razorpay credentials. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET.")


def _basic_auth() -> str:
    pair = f"{settings.razorpay_key_id}:{settings.razorpay_key_secret}".encode("utf-8")
    return base64.b64encode(pair).decode("utf-8")


def _plan_meta(tier: str, billing_cycle: str) -> dict:
    meta = PLAN_MATRIX[(tier, billing_cycle)].copy()
    plan_id = os.getenv(meta["plan_id_env"])
    if not plan_id:
        raise RuntimeError(
            f"Missing Razorpay plan id env var {meta['plan_id_env']}. "
            "Create separate Razorpay plans for each tier and billing cycle."
        )
    meta["plan_id"] = plan_id
    return meta
