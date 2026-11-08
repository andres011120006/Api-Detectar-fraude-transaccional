"""Unit tests for the FraudModel singleton and inference logic."""

import pytest

from src.model import FEATURE_COLUMNS, FraudModel, ModelNotAvailableError
from src.utils import risk_to_status

SAFE = {
    "amount": 25.5,
    "hour_of_day": 14,
    "is_foreign_card": False,
    "failed_attempts_last_24h": 0,
    "distance_from_home_km": 2.0,
    "transaction_velocity_1h": 1,
}
RISKY = {
    "amount": 4500.0,
    "hour_of_day": 3,
    "is_foreign_card": True,
    "failed_attempts_last_24h": 6,
    "distance_from_home_km": 800.0,
    "transaction_velocity_1h": 9,
}


def test_singleton_returns_same_instance():
    assert FraudModel() is FraudModel()


def test_scores_are_probabilities():
    for features in (SAFE, RISKY):
        score = FraudModel().predict_proba(features)
        assert 0.0 <= score <= 1.0


def test_risky_scores_higher_than_safe():
    model = FraudModel()
    assert model.predict_proba(RISKY) > model.predict_proba(SAFE)


def test_missing_feature_raises_value_error():
    incomplete = {k: v for k, v in SAFE.items() if k != FEATURE_COLUMNS[0]}
    with pytest.raises(ValueError, match="Missing feature"):
        FraudModel().predict_proba(incomplete)


def test_unloaded_model_raises(model_path):
    FraudModel.reset()
    try:
        with pytest.raises(ModelNotAvailableError):
            FraudModel().predict_proba(SAFE)
        with pytest.raises(ModelNotAvailableError):
            FraudModel().ensure_loaded(model_path.parent / "missing.joblib")
    finally:
        FraudModel().load(model_path)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.05, "APPROVED"),
        (0.40, "FLAGGED"),
        (0.79, "FLAGGED"),
        (0.80, "REJECTED"),
        (0.99, "REJECTED"),
    ],
)
def test_risk_to_status(score, expected):
    assert risk_to_status(score, 0.80, 0.40) == expected
