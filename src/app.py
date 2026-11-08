"""FastAPI entrypoint: middlewares, routes and the AWS Lambda (Mangum) handler."""

import logging
import secrets
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum

from src.config import Settings, get_settings
from src.model import FraudModel, ModelNotAvailableError
from src.schemas import (
    HealthCheckResponse,
    InferenceParams,
    TransactionRequest,
    TransactionResponse,
)
from src.utils import risk_to_status, setup_logging

setup_logging(get_settings().log_level)
logger = logging.getLogger("fraud_api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm up the model at startup (cold start friendly)."""
    settings = get_settings()
    try:
        FraudModel().ensure_loaded(settings.model_path)
        logger.info("model_loaded", extra={"model_path": settings.model_path})
    except ModelNotAvailableError as exc:
        logger.error("model_unavailable", extra={"reason": str(exc)})
    yield


app = FastAPI(
    title=get_settings().app_name,
    version=get_settings().app_version,
    description="Real-time transactional fraud detection powered by scikit-learn.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Middlewares (the last one added is the outermost one)
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def process_time_middleware(request: Request, call_next):
    """Adds X-Process-Time / X-Request-ID headers, logs every request, catches 500s."""
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "unhandled_exception",
            extra={"request_id": request_id, "path": request.url.path, "method": request.method},
        )
        response = JSONResponse(
            status_code=500, content={"detail": "Internal server error", "request_id": request_id}
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(elapsed_ms, 2),
        },
    )
    return response


_origins = get_settings().cors_origin_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials="*" not in _origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"],
)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """400 for malformed JSON bodies, 422 for schema violations."""
    errors = exc.errors()
    malformed = any(error.get("type") == "json_invalid" for error in errors)
    return JSONResponse(
        status_code=400 if malformed else 422,
        content={"detail": jsonable_encoder(errors, custom_encoder={Exception: str})},
    )


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
SettingsDep = Annotated[Settings, Depends(get_settings)]


def verify_api_key(
    settings: SettingsDep, x_api_key: Annotated[str | None, Header()] = None
) -> None:
    """Enforces the X-API-Key header only when API_KEY is configured."""
    if not settings.api_key:
        return
    if x_api_key is None or not secrets.compare_digest(
        x_api_key.encode(), settings.api_key.encode()
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_model=HealthCheckResponse, tags=["health"])
def healthcheck(settings: SettingsDep) -> HealthCheckResponse:
    loaded = FraudModel().is_loaded
    return HealthCheckResponse(
        status="ok" if loaded else "degraded",
        version=settings.app_version,
        model_loaded=loaded,
    )


@app.post(
    "/v1/predict",
    response_model=TransactionResponse,
    tags=["fraud"],
    dependencies=[Depends(verify_api_key)],
    responses={
        400: {"description": "Malformed JSON or invalid inference input"},
        401: {"description": "Invalid or missing API key"},
        422: {"description": "Payload failed schema validation"},
        500: {"description": "Unexpected server error"},
        503: {"description": "Model not available"},
    },
)
def predict(transaction: TransactionRequest, settings: SettingsDep) -> TransactionResponse:
    """Score a transaction and return an APPROVED / FLAGGED / REJECTED decision."""
    model = FraudModel()
    try:
        model.ensure_loaded(settings.model_path)
        score = model.predict_proba(transaction.model_dump())
    except ModelNotAvailableError as exc:
        logger.error("model_unavailable", extra={"reason": str(exc)})
        raise HTTPException(status_code=503, detail="Model not available") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    decision = risk_to_status(score, settings.risk_threshold, settings.flag_threshold)
    response = TransactionResponse(
        risk_score=round(score, 6),
        is_fraud=decision == "REJECTED",
        status=decision,
        timestamp=datetime.now(UTC),
        inference_params=InferenceParams(
            model_type=model.model_type,
            model_version=settings.app_version,
            risk_threshold=settings.risk_threshold,
            flag_threshold=settings.flag_threshold,
        ),
    )
    logger.info(
        "transaction_scored",
        extra={
            "transaction_id": str(response.transaction_id),
            "risk_score": response.risk_score,
            "status": response.status,
        },
    )
    return response


# AWS Lambda entrypoint (API Gateway / Function URL): handler = src.app.handler
handler = Mangum(app, lifespan="auto")
