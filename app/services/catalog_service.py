from app.config import settings
from app.models import DesignPreset, FeatureItem, LifetimeOffer, SubscriptionPlan
from app.services.storage_service import count_lifetime_claims


def get_subscription_catalog() -> list[SubscriptionPlan]:
    return [
        SubscriptionPlan(
            slug="student",
            name="Student",
            audience="Individuals creating polished decks regularly",
            price_monthly=299,
            price_annual=2999,
            savings_label="Save Rs 589 yearly",
            description="For students, founders, and independent professionals who want polished decks at a price clearly below Gamma Plus.",
            cta="Start student plan",
            features=[
                FeatureItem(label="Best for light but recurring usage"),
                FeatureItem(label="Up to 15 slides per generation"),
                FeatureItem(label="Presentation-ready PPTX export", emphasis=True),
                FeatureItem(label="Slide preview workflow before export"),
                FeatureItem(label="PPTX plus optional PDF export"),
            ],
        ),
        SubscriptionPlan(
            slug="corporate",
            name="Corporate",
            audience="Teams with ongoing deck production",
            price_monthly=799,
            price_annual=7999,
            savings_label="Save Rs 1,589 yearly",
            description="The main plan for startups and teams that need stronger deck throughput while still staying under Gamma Pro pricing.",
            cta="Choose corporate",
            recommended=True,
            features=[
                FeatureItem(label="Everything in Student"),
                FeatureItem(label="Brand-safe design presets", emphasis=True),
                FeatureItem(label="Proposal, QBR, GTM, and sales narratives"),
                FeatureItem(label="Priority rendering and better turnaround"),
                FeatureItem(label="Designed for weekly internal and client usage"),
            ],
        ),
        SubscriptionPlan(
            slug="executive",
            name="Executive",
            audience="Agencies, leaders, and API-heavy customers",
            price_monthly=1799,
            price_annual=17999,
            savings_label="Save Rs 3,589 yearly",
            description="Premium pricing for agencies, leadership teams, and customers who need higher throughput, deeper polish, or API-led workflows.",
            cta="Go executive",
            features=[
                FeatureItem(label="Everything in Corporate"),
                FeatureItem(label="Executive narrative tuning", emphasis=True),
                FeatureItem(label="Advanced polish and premium queue"),
                FeatureItem(label="Best fit for API resale and higher-volume usage"),
                FeatureItem(label="Dedicated support and roadmap priority"),
            ],
        ),
    ]


def get_lifetime_offer() -> LifetimeOffer:
    claimed_spots = count_lifetime_claims()
    remaining_spots = max(settings.razorpay_lifetime_limit - claimed_spots, 0)
    return LifetimeOffer(
        slug="earlybird",
        name="Earlybird Lifetime",
        price=settings.razorpay_lifetime_amount // 100,
        original_price=9999,
        description="One-time launch offer for the first 1000 believers. Pay once and keep core creator access for life before the offer disappears.",
        cta="Claim lifetime deal",
        badge="Limited launch offer",
        note="Once the first 1000 lifetime customers are in, this launch offer closes for good.",
        claimed_spots=claimed_spots,
        remaining_spots=remaining_spots,
        sold_out=remaining_spots == 0,
        features=[
            FeatureItem(label="One-time payment, no recurring subscription", emphasis=True),
            FeatureItem(label="Lifetime access to current creator features"),
            FeatureItem(label="Priority access to future API beta programs"),
            FeatureItem(label="Ideal for creating urgency and social proof"),
        ],
    )


def get_design_catalog() -> list[DesignPreset]:
    return [
        DesignPreset(id=1, name="Midnight Boardroom", vibe="Dark, premium, executive", accent="#68d5ff", best_for="Strategy, fundraising, board reviews"),
        DesignPreset(id=2, name="Signal Orange", vibe="Bold, modern, startup", accent="#ff7a29", best_for="Product launches, investor updates"),
        DesignPreset(id=3, name="Forest Ledger", vibe="Editorial, grounded, trustworthy", accent="#33b36b", best_for="Consulting, operations, ESG"),
        DesignPreset(id=4, name="Crimson Insight", vibe="Sharp, data-first, high contrast", accent="#ff5a67", best_for="Analytics, sales, finance"),
        DesignPreset(id=5, name="Ocean Atlas", vibe="Clean, spacious, polished", accent="#3c82f6", best_for="Client decks, education, explainers"),
        DesignPreset(id=6, name="Royal Copper", vibe="Luxury, premium, persuasive", accent="#f0a85c", best_for="Agency, executive, premium proposals"),
    ]
