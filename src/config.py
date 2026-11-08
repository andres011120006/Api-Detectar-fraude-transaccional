"""Application settings loaded from environment variables (12-factor)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(BASE_DIR / "model.joblib")


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden via env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "Fraud Detection API"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # Path to the serialized scikit-learn model (MODEL_PATH).
    model_path: str = DEFAULT_MODEL_PATH

    # score >= risk_threshold  -> REJECTED (is_fraud = True)
    # score >= flag_threshold  -> FLAGGED  (manual review)
    risk_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    flag_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

    # If set (API_KEY), /v1/predict requires the X-API-Key header.
    api_key: str | None = None

    # Comma separated list of allowed origins, "*" allows all.
    cors_origins: str = "*"

    @model_validator(mode="after")
    def _check_thresholds(self) -> "Settings":
        if self.flag_threshold > self.risk_threshold:
            raise ValueError("FLAG_THRESHOLD must be <= RISK_THRESHOLD")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
