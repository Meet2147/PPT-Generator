from app.models import DesignPreset, FeatureItem, SubscriptionPlan


def get_subscription_catalog() -> list[SubscriptionPlan]:
    return [
        SubscriptionPlan(
            slug="student",
            name="Student",
            audience="Students, founders, solo operators",
            price_monthly=8,
            price_annual=79,
            savings_label="Save 18% yearly",
            description="Fast, polished decks for coursework, pitches, and internships.",
            cta="Start student plan",
            features=[
                FeatureItem(label="Unlimited presentations"),
                FeatureItem(label="Up to 15 slides per generation"),
                FeatureItem(label="Presentation-ready PPTX export", emphasis=True),
                FeatureItem(label="Academic and startup deck presets"),
                FeatureItem(label="Email support"),
            ],
        ),
        SubscriptionPlan(
            slug="corporate",
            name="Corporate",
            audience="Teams building sales, strategy, and client decks",
            price_monthly=24,
            price_annual=228,
            savings_label="Save 21% yearly",
            description="Designed to undercut Gamma-style business plans with cleaner PPT output.",
            cta="Choose corporate",
            recommended=True,
            features=[
                FeatureItem(label="Everything in Student"),
                FeatureItem(label="Brand-safe design presets", emphasis=True),
                FeatureItem(label="Meeting, proposal, and QBR narratives"),
                FeatureItem(label="Priority rendering queue"),
                FeatureItem(label="Shared workspace roadmap"),
            ],
        ),
        SubscriptionPlan(
            slug="executive",
            name="Executive",
            audience="Leadership, agencies, and premium client delivery",
            price_monthly=49,
            price_annual=468,
            savings_label="Save 20% yearly",
            description="Premium tier for high-stakes decks, polished storytelling, and concierge support.",
            cta="Go executive",
            features=[
                FeatureItem(label="Everything in Corporate"),
                FeatureItem(label="Executive narrative tuning", emphasis=True),
                FeatureItem(label="Advanced visual polish"),
                FeatureItem(label="White-label export roadmap"),
                FeatureItem(label="Dedicated support"),
            ],
        ),
    ]


def get_design_catalog() -> list[DesignPreset]:
    return [
        DesignPreset(id=1, name="Midnight Boardroom", vibe="Dark, premium, executive", accent="#68d5ff", best_for="Strategy, fundraising, board reviews"),
        DesignPreset(id=2, name="Signal Orange", vibe="Bold, modern, startup", accent="#ff7a29", best_for="Product launches, investor updates"),
        DesignPreset(id=3, name="Forest Ledger", vibe="Editorial, grounded, trustworthy", accent="#33b36b", best_for="Consulting, operations, ESG"),
        DesignPreset(id=4, name="Crimson Insight", vibe="Sharp, data-first, high contrast", accent="#ff5a67", best_for="Analytics, sales, finance"),
        DesignPreset(id=5, name="Ocean Atlas", vibe="Clean, spacious, polished", accent="#3c82f6", best_for="Client decks, education, explainers"),
        DesignPreset(id=6, name="Royal Copper", vibe="Luxury, premium, persuasive", accent="#f0a85c", best_for="Agency, executive, premium proposals"),
    ]
