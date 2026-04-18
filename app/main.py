from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models import (
    BillingCheckoutRequest,
    BillingCheckoutResponse,
    BillingWebhookResponse,
    GenerationRequest,
    GenerationResponse,
)
from app.services.ai_service import build_deck_draft, summarize_deck
from app.services.billing_service import (
    create_checkout_subscription,
    persist_webhook_event,
    verify_webhook_signature,
)
from app.services.catalog_service import get_design_catalog, get_subscription_catalog
from app.services.deck_service import build_presentation


settings.generated_dir.mkdir(parents=True, exist_ok=True)
settings.cache_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="DeckMint API",
    version="2.0.0",
    summary="Subscription-first PowerPoint generation API and web app.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
async def health() -> dict:
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


@app.get("/api/v1/catalog")
async def catalog() -> dict:
    return {
        "product": {
            "name": settings.app_name,
            "tagline": "Generate polished, editable presentations with subscription pricing.",
        },
        "plans": [plan.model_dump() for plan in get_subscription_catalog()],
        "designs": [design.model_dump() for design in get_design_catalog()],
    }


@app.post("/api/v1/presentations/generate", response_model=GenerationResponse)
async def generate_presentation(payload: GenerationRequest) -> GenerationResponse:
    deck = build_deck_draft(payload)
    file_path = await build_presentation(payload, deck)
    public_base = settings.public_base_url.rstrip("/")
    return GenerationResponse(
        status="success",
        title=deck.title,
        filename=file_path.name,
        download_url=f"{public_base}/api/v1/presentations/download/{file_path.name}",
        slide_count=len(deck.slides),
        fallback_used=deck.fallback_used,
        summary=summarize_deck(deck),
    )


@app.post("/api/v1/billing/checkout", response_model=BillingCheckoutResponse)
async def create_billing_checkout(payload: BillingCheckoutRequest) -> BillingCheckoutResponse:
    try:
        checkout = create_checkout_subscription(payload)
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Razorpay error: {detail}")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return BillingCheckoutResponse(**checkout)


@app.post("/api/v1/billing/webhooks/razorpay", response_model=BillingWebhookResponse)
async def razorpay_webhook(request: Request) -> BillingWebhookResponse:
    raw_body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature")
    if not verify_webhook_signature(raw_body, signature):
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")

    persist_webhook_event(raw_body)
    event = "unknown"
    try:
        payload = await request.json()
        event = payload.get("event", "unknown")
    except Exception:
        pass
    return BillingWebhookResponse(status="ok", event=event)


@app.get("/api/v1/presentations/download/{filename}")
async def download_presentation(filename: str):
    safe_name = Path(filename).name
    file_path = settings.generated_dir / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Presentation not found.")
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=file_path.name,
    )


if settings.static_dir.exists():
    app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="web")
