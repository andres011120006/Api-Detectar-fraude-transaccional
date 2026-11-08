"""Shared fixtures: trains a tiny model once per session so tests are self-contained."""

import joblib
import pytest
from fastapi.testclient import TestClient

from data.generate_dataset import generate_dataset
from src.app import app
from src.model import FraudModel
from train import train_model


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    dataset = generate_dataset(n_samples=4000, seed=7)
    model, _ = train_model(dataset, n_estimators=60, seed=7)
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(model, path)
    return path


@pytest.fixture(scope="session", autouse=True)
def loaded_model(model_path):
    FraudModel.reset()
    FraudModel().load(model_path)
    yield FraudModel()
    FraudModel.reset()


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client
