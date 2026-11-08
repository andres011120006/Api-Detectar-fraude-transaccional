"""Synthetic (but realistic) transactional dataset generator.

Usage:
    python data/generate_dataset.py --samples 6000 --fraud-ratio 0.08 --output data/transactions.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "amount",
    "hour_of_day",
    "is_foreign_card",
    "failed_attempts_last_24h",
    "distance_from_home_km",
    "transaction_velocity_1h",
]
TARGET = "is_fraud"

# Legitimate traffic peaks during the day; fraud skews towards the night.
_LEGIT_HOUR_WEIGHTS = np.array(
    [1, 0.6, 0.4, 0.3, 0.3, 0.6, 1.5, 3, 4, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1.5]
)
_FRAUD_HOUR_WEIGHTS = np.array(
    [5, 6, 6, 6, 5, 4, 1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 2, 2, 3, 4]
)
_LEGIT_HOUR_P = _LEGIT_HOUR_WEIGHTS / _LEGIT_HOUR_WEIGHTS.sum()
_FRAUD_HOUR_P = _FRAUD_HOUR_WEIGHTS / _FRAUD_HOUR_WEIGHTS.sum()


def _legit_transactions(rng: np.random.Generator, n: int) -> pd.DataFrame:
    distance = rng.exponential(10, n)
    traveling = rng.random(n) < 0.05
    distance[traveling] += rng.exponential(150, traveling.sum())
    return pd.DataFrame(
        {
            "amount": np.clip(rng.lognormal(3.6, 0.9, n), 1, 20_000).round(2),
            "hour_of_day": rng.choice(24, size=n, p=_LEGIT_HOUR_P),
            "is_foreign_card": (rng.random(n) < 0.04).astype(int),
            "failed_attempts_last_24h": rng.poisson(0.15, n),
            "distance_from_home_km": np.clip(distance, 0, 20_000).round(2),
            "transaction_velocity_1h": rng.poisson(0.8, n),
            TARGET: 0,
        }
    )


def _fraud_transactions(rng: np.random.Generator, n: int) -> pd.DataFrame:
    card_testing = rng.random(n) < 0.30  # many small charges in a short time
    amount = np.where(card_testing, rng.lognormal(2.5, 0.6, n), rng.lognormal(6.0, 1.0, n))
    velocity = rng.poisson(4.5, n) + np.where(card_testing, rng.poisson(3, n), 0)
    frame = pd.DataFrame(
        {
            "amount": np.clip(amount, 1, 50_000).round(2),
            "hour_of_day": rng.choice(24, size=n, p=_FRAUD_HOUR_P),
            "is_foreign_card": (rng.random(n) < 0.60).astype(int),
            "failed_attempts_last_24h": rng.poisson(2.5, n),
            "distance_from_home_km": np.clip(rng.exponential(250, n) + 20, 0, 20_000).round(2),
            "transaction_velocity_1h": velocity,
            TARGET: 1,
        }
    )
    # "Stealth" fraud: ~12% of frauds mimic normal behaviour to keep the problem realistic.
    stealth = rng.random(n) < 0.12
    k = int(stealth.sum())
    if k:
        frame.loc[stealth, "hour_of_day"] = rng.choice(24, size=k, p=_LEGIT_HOUR_P)
        frame.loc[stealth, "failed_attempts_last_24h"] = rng.poisson(0.3, k)
        frame.loc[stealth, "transaction_velocity_1h"] = rng.poisson(1.0, k)
    return frame


def generate_dataset(
    n_samples: int = 6000, fraud_ratio: float = 0.08, seed: int = 42
) -> pd.DataFrame:
    """Return a shuffled DataFrame with COLUMNS + `is_fraud` (imbalanced by default)."""
    if n_samples < 100:
        raise ValueError("n_samples must be >= 100")
    if not 0.0 < fraud_ratio < 1.0:
        raise ValueError("fraud_ratio must be between 0 and 1 (exclusive)")

    rng = np.random.default_rng(seed)
    n_fraud = max(1, round(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud
    data = pd.concat(
        [_legit_transactions(rng, n_legit), _fraud_transactions(rng, n_fraud)], ignore_index=True
    )
    data = data.iloc[rng.permutation(len(data))].reset_index(drop=True)
    return data[[*COLUMNS, TARGET]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic fraud dataset.")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--fraud-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "transactions.csv"
    )
    args = parser.parse_args()

    dataset = generate_dataset(args.samples, args.fraud_ratio, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Saved {len(dataset)} rows ({dataset[TARGET].mean():.2%} fraud) -> {args.output}")


if __name__ == "__main__":
    main()
