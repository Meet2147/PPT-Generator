import base64
import hashlib
import hmac
import time
import requests

from app.config import settings
from app.models import BillingCheckoutRequest
from app.services.storage_service import count_lifetime_claims


PLAN_MATRIX = {
    ("student", "monthly"): {
        "settings_key": "razorpay_plan_student_monthly",
        "plan_name": "Student Monthly",
        "description": "Student plan billed monthly",
        "amount": 29900,
    },
    ("student", "annual"): {
        "settings_key": "razorpay_plan_student_annual",
        "plan_name": "Student Annual",
        "description": "Student plan billed annually",
        "amount": 299900,
    },
    ("corporate", "monthly"): {
        "settings_key": "razorpay_plan_corporate_monthly",
        "plan_name": "Corporate Monthly",
        "description": "Corporate plan billed monthly",
        "amount": 79900,
    },
    ("corporate", "annual"): {
        "settings_key": "razorpay_plan_corporate_annual",
        "plan_name": "Corporate Annual",
        "description": "Corporate plan billed annually",
        "amount": 799900,
    },
    ("executive", "monthly"): {
        "settings_key": "razorpay_plan_executive_monthly",
        "plan_name": "Executive Monthly",
        "description": "Executive plan billed monthly",
        "amount": 179900,
    },
    ("executive", "annual"): {
        "settings_key": "razorpay_plan_executive_annual",
        "plan_name": "Executive Annual",
        "description": "Executive plan billed annually",
        "amount": 1799900,
    },
}


def create_checkout_subscription(payload: BillingCheckoutRequest) -> dict:
    _require_razorpay_config()
    if payload.billing_cycle == "lifetime" or payload.tier == "earlybird":
        return create_lifetime_payment_link(payload)

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
        "flow_type": "subscription",
        "key_id": settings.razorpay_key_id,
        "subscription_id": subscription["id"],
        "plan_id": meta["plan_id"],
        "checkout_url": None,
        "amount": meta["amount"],
        "currency": "INR",
        "plan_name": meta["plan_name"],
        "description": meta["description"],
        "customer": {
            "name": payload.name or "",
            "email": payload.email or "",
        },
    }


def create_lifetime_payment_link(payload: BillingCheckoutRequest) -> dict:
    if count_lifetime_claims() >= settings.razorpay_lifetime_limit:
        raise RuntimeError("The earlybird lifetime offer is sold out.")
    amount = settings.razorpay_lifetime_amount
    response = requests.post(
        "https://api.razorpay.com/v1/payment_links",
        headers={
            "Authorization": f"Basic {_basic_auth()}",
            "Content-Type": "application/json",
        },
        json={
            "amount": amount,
            "currency": "INR",
            "description": "DeckMint Earlybird Lifetime access",
            "reference_id": f"deckmint-lifetime-{int(time.time())}",
            "customer": {
                "name": payload.name or "",
                "email": payload.email or "",
            },
            "notify": {
                "email": bool(payload.email),
                "sms": False,
            },
            "notes": {
                "tier": "earlybird",
                "billing_cycle": "lifetime",
                "product": settings.app_name,
            },
            "callback_url": f"{settings.web_base_url.rstrip('/')}/#pricing",
            "callback_method": "get",
        },
        timeout=45,
    )
    response.raise_for_status()
    payment_link = response.json()
    return {
        "status": "created",
        "provider": "razorpay",
        "flow_type": "payment_link",
        "key_id": None,
        "subscription_id": None,
        "plan_id": None,
        "checkout_url": payment_link.get("short_url") or payment_link.get("payment_link_url"),
        "amount": amount,
        "currency": "INR",
        "plan_name": "Earlybird Lifetime",
        "description": "One-time launch offer for lifetime creator access",
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
    plan_id = getattr(settings, meta["settings_key"], None)
    if not plan_id:
        raise RuntimeError(
            f"Missing Razorpay plan id setting {meta['settings_key']}. "
            "Create separate Razorpay plans for each tier and billing cycle."
        )
    meta["plan_id"] = plan_id
    return meta
