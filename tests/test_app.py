"""HTTP-level tests for the FastAPI application."""

import pytest

from src.app import app
from src.config import Settings, get_settings

SAFE_TRANSACTION = {
    "amount": 25.5,
    "hour_of_day": 14,
    "is_foreign_card": False,
    "failed_attempts_last_24h": 0,
    "distance_from_home_km": 2.0,
    "transaction_velocity_1h": 1,
}

RISKY_TRANSACTION = {
    "amount": 4500.0,
    "hour_of_day": 3,
    "is_foreign_card": True,
    "failed_attempts_last_24h": 6,
    "distance_from_home_km": 800.0,
    "transaction_velocity_1h": 9,
}


def test_healthcheck_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["model_loaded"] is True


def test_safe_transaction_is_approved(client):
    response = client.post("/v1/predict", json=SAFE_TRANSACTION)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "APPROVED"
    assert body["is_fraud"] is False
    assert 0.0 <= body["risk_score"] < body["inference_params"]["flag_threshold"]


def test_risky_transaction_is_rejected_or_flagged(client):
    response = client.post("/v1/predict", json=RISKY_TRANSACTION)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"REJECTED", "FLAGGED"}
    assert body["risk_score"] > 0.4


def test_response_contract_and_headers(client):
    response = client.post("/v1/predict", json=SAFE_TRANSACTION)
    body = response.json()
    assert len(body["transaction_id"]) == 36
    assert body["timestamp"]
    assert {"model_type", "model_version", "risk_threshold", "flag_threshold"} <= set(
        body["inference_params"]
    )
    assert response.headers["X-Process-Time"].endswith("ms")
    assert response.headers["X-Request-ID"]


@pytest.mark.parametrize(
    "payload",
    [
        {**SAFE_TRANSACTION, "amount": -10},
        {**SAFE_TRANSACTION, "hour_of_day": 25},
        {**SAFE_TRANSACTION, "is_foreign_card": "maybe"},
        {**SAFE_TRANSACTION, "unexpected_field": 1},
        {"amount": 10},
        {},
    ],
)
def test_invalid_payload_returns_422(client, payload):
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_malformed_json_returns_400(client):
    response = client.post(
        "/v1/predict", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400


def test_api_key_is_enforced_when_configured(client):
    app.dependency_overrides[get_settings] = lambda: Settings(api_key="secret-key")
    try:
        assert client.post("/v1/predict", json=SAFE_TRANSACTION).status_code == 401
        wrong = client.post("/v1/predict", json=SAFE_TRANSACTION, headers={"X-API-Key": "nope"})
        assert wrong.status_code == 401
        ok = client.post("/v1/predict", json=SAFE_TRANSACTION, headers={"X-API-Key": "secret-key"})
        assert ok.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_unexpected_error_returns_500(client, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.app.FraudModel.predict_proba", boom)
    response = client.post("/v1/predict", json=SAFE_TRANSACTION)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
