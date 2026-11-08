"""Pydantic v2 request/response contracts."""

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """Transaction features sent by the client. Strict types, no extra fields."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "amount": 1250.75,
                    "hour_of_day": 3,
                    "is_foreign_card": True,
                    "failed_attempts_last_24h": 4,
                    "distance_from_home_km": 640.0,
                    "transaction_velocity_1h": 7,
                }
            ]
        },
    )

    amount: float = Field(
        ..., gt=0, le=1_000_000, allow_inf_nan=False, description="Transaction amount in USD."
    )
    hour_of_day: int = Field(..., ge=0, le=23, description="Local hour of the transaction (0-23).")
    is_foreign_card: bool = Field(..., description="True if the card was issued abroad.")
    failed_attempts_last_24h: int = Field(
        ..., ge=0, le=100, description="Failed payment attempts in the last 24 hours."
    )
    distance_from_home_km: float = Field(
        ...,
        ge=0,
        le=20_000,
        allow_inf_nan=False,
        description="Distance between transaction location and cardholder home (km).",
    )
    transaction_velocity_1h: int = Field(
        ..., ge=0, le=500, description="Number of transactions by the card in the last hour."
    )


class InferenceParams(BaseModel):
    """Parameters used for this inference (traceability / auditing)."""

    model_config = ConfigDict(protected_namespaces=())

    model_type: str = Field(..., description="Estimator class used for scoring.")
    model_version: str = Field(..., description="Application/model version.")
    risk_threshold: float = Field(..., description="score >= this -> REJECTED.")
    flag_threshold: float = Field(..., description="score >= this -> FLAGGED.")


class TransactionResponse(BaseModel):
    transaction_id: UUID = Field(default_factory=uuid4, description="Unique UUIDv4 identifier.")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Fraud probability in [0, 1].")
    is_fraud: bool = Field(..., description="True when the transaction is rejected as fraud.")
    status: Literal["APPROVED", "FLAGGED", "REJECTED"]
    timestamp: datetime = Field(..., description="UTC time of the decision.")
    inference_params: InferenceParams


class HealthCheckResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
