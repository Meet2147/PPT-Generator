from pathlib import Path

import requests
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models import (
    AuthResponse,
    BillingCheckoutRequest,
    BillingCheckoutResponse,
    BillingWebhookResponse,
    DeckHistoryItem,
    DeckPreviewResponse,
    DocsTokenRequest,
    DocsTokenResponse,
    GenerationRequest,
    GenerationResponse,
    LoginRequest,
    SignupRequest,
    UserResponse,
)
from app.services.ai_service import build_deck_draft, summarize_deck
from app.services.auth_service import issue_auth_token, login_user, require_user_from_token, signup_user
from app.services.billing_service import (
    create_checkout_subscription,
    persist_webhook_event,
    verify_webhook_signature,
)
from app.services.catalog_service import get_design_catalog, get_lifetime_offer, get_subscription_catalog
from app.services.deck_service import build_file_stem, build_preview_slides, build_presentation
from app.services.docs_auth_service import issue_docs_token, require_docs_admin, verify_docs_request
from app.services.export_service import build_pdf_export
from app.services.storage_service import init_db, list_user_decks, record_deck_history, record_lifetime_claim


settings.generated_dir.mkdir(parents=True, exist_ok=True)
settings.cache_dir.mkdir(parents=True, exist_ok=True)
init_db()

app = FastAPI(
    title="DeckMint API",
    version="2.0.0",
    summary="Subscription-first PowerPoint generation API and web app.",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> UserResponse:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return require_user_from_token(credentials.credentials)


@app.get("/api/v1/health")
async def health() -> dict:
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


@app.post("/api/v1/auth/signup", response_model=AuthResponse)
async def auth_signup(payload: SignupRequest) -> AuthResponse:
    user = signup_user(payload.name, payload.email, payload.password)
    return AuthResponse(status="success", access_token=issue_auth_token(user), user=user)


@app.post("/api/v1/auth/login", response_model=AuthResponse)
async def auth_login(payload: LoginRequest) -> AuthResponse:
    user = login_user(payload.email, payload.password)
    return AuthResponse(status="success", access_token=issue_auth_token(user), user=user)


@app.get("/api/v1/auth/me", response_model=UserResponse)
async def auth_me(user: UserResponse = Depends(current_user)) -> UserResponse:
    return user


@app.get("/api/v1/decks/history", response_model=list[DeckHistoryItem])
async def deck_history(user: UserResponse = Depends(current_user)) -> list[DeckHistoryItem]:
    public_base = settings.public_base_url.rstrip("/")
    items = []
    for row in list_user_decks(user.id):
        items.append(
            DeckHistoryItem(
                id=row["id"],
                title=row["title"],
                topic=row["topic"],
                audience=row["audience"],
                tone=row["tone"],
                slide_count=row["slide_count"],
                pptx_filename=row["pptx_filename"],
                pptx_download_url=f"{public_base}/api/v1/presentations/download/{row['pptx_filename']}",
                pdf_filename=row["pdf_filename"],
                pdf_download_url=(
                    f"{public_base}/api/v1/presentations/download/{row['pdf_filename']}"
                    if row["pdf_filename"]
                    else None
                ),
                created_at=row["created_at"],
            )
        )
    return items


@app.get("/api/v1/catalog")
async def catalog() -> dict:
    return {
        "product": {
            "name": settings.app_name,
            "tagline": "Generate polished, editable presentations with subscription pricing.",
        },
        "plans": [plan.model_dump() for plan in get_subscription_catalog()],
        "lifetime_offer": get_lifetime_offer().model_dump(),
        "designs": [design.model_dump() for design in get_design_catalog()],
    }


@app.post("/api/v1/docs-token", response_model=DocsTokenResponse)
async def create_docs_token(payload: DocsTokenRequest, request: Request) -> DocsTokenResponse:
    require_docs_admin(request)
    return DocsTokenResponse(**issue_docs_token(payload.requested_by))


@app.post("/api/v1/presentations/preview", response_model=DeckPreviewResponse)
async def preview_presentation(payload: GenerationRequest, user: UserResponse = Depends(current_user)) -> DeckPreviewResponse:
    deck = build_deck_draft(payload)
    preview_slides = build_preview_slides(payload, deck)
    return DeckPreviewResponse(
        title=deck.title,
        subtitle=deck.subtitle,
        slide_count=len(deck.slides),
        fallback_used=deck.fallback_used,
        slides=preview_slides,
        summary=summarize_deck(deck),
    )


@app.post("/api/v1/presentations/generate", response_model=GenerationResponse)
async def generate_presentation(payload: GenerationRequest, user: UserResponse = Depends(current_user)) -> GenerationResponse:
    deck = build_deck_draft(payload)
    file_stem = build_file_stem(deck.title)
    pptx_path = await build_presentation_to_path(payload, deck, file_stem)
    pdf_path = None
    if payload.export_pdf:
        pdf_path = await build_pdf_export(payload, deck, file_stem, settings.generated_dir)

    record_deck_history(
        user_id=user.id,
        title=deck.title,
        topic=payload.topic,
        audience=payload.audience,
        tone=payload.tone,
        slide_count=len(deck.slides),
        pptx_filename=pptx_path.name,
        pdf_filename=pdf_path.name if pdf_path else None,
    )

    public_base = settings.public_base_url.rstrip("/")
    return GenerationResponse(
        status="success",
        title=deck.title,
        filename=pptx_path.name,
        download_url=f"{public_base}/api/v1/presentations/download/{pptx_path.name}",
        slide_count=len(deck.slides),
        fallback_used=deck.fallback_used,
        summary=summarize_deck(deck),
        pdf_filename=pdf_path.name if pdf_path else None,
        pdf_download_url=(
            f"{public_base}/api/v1/presentations/download/{pdf_path.name}" if pdf_path else None
        ),
    )


async def build_presentation_to_path(payload: GenerationRequest, deck, file_stem: str) -> Path:
    path = settings.generated_dir / f"{file_stem}.pptx"
    generated = await build_presentation(payload, deck)
    if generated != path:
        generated.rename(path)
    return path


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
        maybe_record_lifetime_claim(payload)
    except Exception:
        pass
    return BillingWebhookResponse(status="ok", event=event)


def maybe_record_lifetime_claim(payload: dict) -> None:
    payment_link_entity = payload.get("payload", {}).get("payment_link", {}).get("entity", {})
    payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
    notes = payment_link_entity.get("notes") or payment_entity.get("notes") or {}
    if notes.get("tier") != "earlybird":
        return
    record_lifetime_claim(
        razorpay_payment_link_id=payment_link_entity.get("id"),
        razorpay_payment_id=payment_entity.get("id"),
        email=payment_entity.get("email"),
    )


@app.get("/api/v1/presentations/download/{filename}")
async def download_presentation(filename: str):
    safe_name = Path(filename).name
    file_path = settings.generated_dir / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Presentation not found.")
    media_type = (
        "application/pdf"
        if safe_name.lower().endswith(".pdf")
        else "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )
    return FileResponse(file_path, media_type=media_type, filename=file_path.name)


@app.get("/openapi.json", include_in_schema=False)
async def protected_openapi(request: Request):
    verify_docs_request(request)
    return get_openapi(
        title=app.title,
        version=app.version,
        summary=app.summary,
        description=app.description,
        routes=app.routes,
    )


@app.get("/docs", include_in_schema=False)
async def protected_docs(request: Request):
    verify_docs_request(request)
    token = request.query_params.get("token") or ""
    return get_swagger_ui_html(openapi_url=f"/openapi.json?token={token}", title=f"{app.title} - Swagger UI")


if settings.static_dir.exists():
    app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="web")
