"""Thread-safe singleton wrapper around the serialized fraud model."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "amount",
    "hour_of_day",
    "is_foreign_card",
    "failed_attempts_last_24h",
    "distance_from_home_km",
    "transaction_velocity_1h",
]


class ModelNotAvailableError(RuntimeError):
    """Raised when the model file is missing or has not been loaded."""


class FraudModel:
    """Singleton: the .joblib file is loaded once per process, safely across threads."""

    _instance: FraudModel | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> FraudModel:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._estimator = None
                    instance._path = None
                    instance._load_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    _estimator: Any
    _path: Path | None
    _load_lock: threading.Lock

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton (used by tests)."""
        with cls._instance_lock:
            cls._instance = None

    @property
    def is_loaded(self) -> bool:
        return self._estimator is not None

    @property
    def model_type(self) -> str:
        return type(self._estimator).__name__ if self.is_loaded else "unloaded"

    def load(self, path: str | Path) -> None:
        """Load (or reload) the model from disk."""
        model_file = Path(path)
        with self._load_lock:
            if not model_file.is_file():
                raise ModelNotAvailableError(f"Model file not found: {model_file}")
            estimator = joblib.load(model_file)
            if not hasattr(estimator, "predict_proba") or not hasattr(estimator, "classes_"):
                raise ModelNotAvailableError("Loaded object is not a fitted classifier")
            self._estimator = estimator
            self._path = model_file

    def ensure_loaded(self, path: str | Path) -> None:
        """Load the model only if it has not been loaded yet (double-checked locking)."""
        if self.is_loaded:
            return
        with self._load_lock:
            already_loaded = self.is_loaded
        if not already_loaded:
            self.load(path)

    def predict_proba(self, features: Mapping[str, float | int | bool]) -> float:
        """Return the probability (0-1) that the transaction is fraudulent."""
        estimator = self._estimator
        if estimator is None:
            raise ModelNotAvailableError("Model is not loaded")
        try:
            row = [[float(features[column]) for column in FEATURE_COLUMNS]]
        except KeyError as exc:
            raise ValueError(f"Missing feature: {exc.args[0]}") from exc
        frame = pd.DataFrame(row, columns=FEATURE_COLUMNS)
        probabilities = estimator.predict_proba(frame)[0]
        classes = list(estimator.classes_)
        if 1 not in classes:
            return 0.0
        score = float(probabilities[classes.index(1)])
        return min(max(score, 0.0), 1.0)


def get_model() -> FraudModel:
    return FraudModel()
