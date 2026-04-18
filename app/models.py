from typing import Literal

from pydantic import BaseModel, Field


PlanKey = Literal["monthly", "annual"]
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


class GenerationResponse(BaseModel):
    status: str
    title: str
    filename: str
    download_url: str
    slide_count: int
    fallback_used: bool
    summary: str


class BillingCheckoutRequest(BaseModel):
    tier: AudienceKey
    billing_cycle: PlanKey
    email: str | None = None
    name: str | None = None
    quantity: int = Field(default=1, ge=1, le=500)


class BillingCheckoutResponse(BaseModel):
    status: str
    provider: str
    key_id: str
    subscription_id: str
    plan_id: str
    amount: int | None = None
    currency: str = "INR"
    plan_name: str
    description: str
    customer: dict


class BillingWebhookResponse(BaseModel):
    status: str
    event: str
