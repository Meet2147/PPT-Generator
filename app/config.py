from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "DeckMint"
    app_env: str = "dev"
    public_base_url: str = "https://api.dashovia.com"
    web_base_url: str = "https://slides.dashovia.com"
    pplx_api_key: str | None = None
    pplx_model: str = "sonar-pro"
    razorpay_key_id: str | None = None
    razorpay_key_secret: str | None = None
    razorpay_webhook_secret: str | None = None
    razorpay_plan_student_monthly: str | None = None
    razorpay_plan_student_annual: str | None = None
    razorpay_plan_corporate_monthly: str | None = None
    razorpay_plan_corporate_annual: str | None = None
    razorpay_plan_executive_monthly: str | None = None
    razorpay_plan_executive_annual: str | None = None
    razorpay_lifetime_amount: int = 199900
    razorpay_lifetime_limit: int = 1000
    docs_admin_secret: str | None = None
    docs_jwt_secret: str | None = None
    docs_token_expiry_hours: int = 24
    default_slide_count: int = 9
    generated_dir: Path = BASE_DIR / "GeneratedPresentations"
    cache_dir: Path = BASE_DIR / "Cache"
    designs_dir: Path = BASE_DIR / "Designs"
    static_dir: Path = BASE_DIR / "web"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
