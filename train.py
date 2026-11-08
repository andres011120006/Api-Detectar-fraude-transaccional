"""Train, evaluate and serialize the fraud detection model.

Usage:
    python train.py --samples 6000 --fraud-ratio 0.08
    python train.py --data data/transactions.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data.generate_dataset import TARGET, generate_dataset
from src.config import DEFAULT_MODEL_PATH
from src.model import FEATURE_COLUMNS


def train_model(
    dataset: pd.DataFrame, n_estimators: int = 200, seed: int = 42
) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Fit a RandomForest on an 80/20 stratified split and return (model, metrics)."""
    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=seed, stratify=target
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument("--data", type=Path, default=None, help="Optional CSV with the dataset.")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--fraud-ratio", type=float, default=0.08)
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    if args.data:
        dataset = pd.read_csv(args.data)
    else:
        dataset = generate_dataset(args.samples, args.fraud_ratio, args.seed)
    print(f"Dataset: {len(dataset)} rows, fraud rate {dataset[TARGET].mean():.2%}")

    model, metrics = train_model(dataset, args.estimators, args.seed)

    # Detailed report on the same hold-out split used for the metrics.
    _, x_test, _, y_test = train_test_split(
        dataset[FEATURE_COLUMNS],
        dataset[TARGET],
        test_size=0.20,
        random_state=args.seed,
        stratify=dataset[TARGET],
    )
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, y_pred))
    print("Metrics:", json.dumps(metrics, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output, compress=3)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
