from typing import Literal

from pydantic import BaseModel, Field


PlanKey = Literal["monthly", "annual"]
BillingCycleKey = Literal["monthly", "annual", "lifetime"]
AudienceKey = Literal["student", "corporate", "executive"]
ToneKey = Literal["professional", "persuasive", "analytical", "visionary"]


class FeatureItem(BaseModel):
    label: str
    emphasis: bool = False


class SubscriptionPlan(BaseModel):
    slug: AudienceKey
    name: str
    audience: str
    price_monthly: int
    price_annual: int
    savings_label: str
    description: str
    cta: str
    recommended: bool = False
    features: list[FeatureItem]


class LifetimeOffer(BaseModel):
    slug: str
    name: str
    price: int
    original_price: int
    description: str
    cta: str
    badge: str
    note: str
    claimed_spots: int
    remaining_spots: int
    sold_out: bool
    features: list[FeatureItem]


class DesignPreset(BaseModel):
    id: int
    name: str
    vibe: str
    accent: str
    best_for: str


class GenerationRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=180)
    audience: AudienceKey = "corporate"
    objective: str | None = Field(default=None, max_length=240)
    tone: ToneKey = "professional"
    plan: PlanKey = "monthly"
    design_number: int = Field(default=2, ge=1, le=6)
    slide_count: int = Field(default=9, ge=6, le=15)
    include_images: bool = True
    export_pdf: bool = False


class SlideDraft(BaseModel):
    number: int
    title: str
    content: str
    image_prompt: str | None = None


class DeckDraft(BaseModel):
    title: str
    subtitle: str
    slides: list[SlideDraft]
    fallback_used: bool = False


class PreviewSlide(BaseModel):
    number: int
    title: str
    content: str
    accent: str
    vibe: str


class DeckPreviewResponse(BaseModel):
    title: str
    subtitle: str
    slide_count: int
    fallback_used: bool
    slides: list[PreviewSlide]
    summary: str


class GenerationResponse(BaseModel):
    status: str
    title: str
    filename: str
    download_url: str
    slide_count: int
    fallback_used: bool
    summary: str
    pdf_filename: str | None = None
    pdf_download_url: str | None = None


class BillingCheckoutRequest(BaseModel):
    tier: AudienceKey | Literal["earlybird"]
    billing_cycle: BillingCycleKey
    email: str | None = None
    name: str | None = None
    quantity: int = Field(default=1, ge=1, le=500)


class BillingCheckoutResponse(BaseModel):
    status: str
    provider: str
    flow_type: Literal["subscription", "payment_link"]
    key_id: str | None = None
    subscription_id: str | None = None
    plan_id: str | None = None
    checkout_url: str | None = None
    amount: int | None = None
    currency: str = "INR"
    plan_name: str
    description: str
    customer: dict


class BillingWebhookResponse(BaseModel):
    status: str
    event: str


class DocsTokenRequest(BaseModel):
    requested_by: str | None = None


class DocsTokenResponse(BaseModel):
    status: str
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    docs_url: str


class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=5, max_length=200)
    password: str = Field(min_length=8, max_length=200)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=200)
    password: str = Field(min_length=8, max_length=200)


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    created_at: str


class AuthResponse(BaseModel):
    status: str
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class DeckHistoryItem(BaseModel):
    id: int
    title: str
    topic: str
    audience: str
    tone: str
    slide_count: int
    pptx_filename: str
    pptx_download_url: str
    pdf_filename: str | None = None
    pdf_download_url: str | None = None
    created_at: str
