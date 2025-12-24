from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "dev"

    gemini_api_key: str
    pplx_api_key: str
    public_base_url:str

    gemini_classifier_model: str = "gemini-2.0-flash"
    gemini_portion_model: str = "gemini-2.0-flash"
    pplx_sonar_model: str = "sonar-pro"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings()