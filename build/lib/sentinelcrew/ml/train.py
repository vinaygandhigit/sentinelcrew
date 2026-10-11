"""Train the Triage baseline model on *normal* behaviour.

Isolation Forest is unsupervised: it learns what normal looks like and flags
deviations. The MVP trains on a reproducible synthetic baseline that mirrors
the distribution of benign enterprise activity (business-hours logins, the
occasional typo'd password, modest egress, few ports). Replace
``generate_baseline`` with features computed from your own historical,
known-good telemetry for a production deployment.

Run:  ``python -m sentinelcrew.ml.train``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sentinelcrew.ml.features import FEATURE_NAMES
from sentinelcrew.ml.isolation_forest import IsolationForest


def generate_baseline(n: int = 2000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    failed = rng.poisson(0.6, n).clip(0, 4)
    distinct_users = rng.choice([0, 1, 1, 1, 2], n)
    success_after = (rng.random(n) < 0.01).astype(float)
    off_hours = (rng.random(n) < 0.12).astype(float)
    external = (rng.random(n) < 0.08).astype(float)
    mb_out = rng.gamma(1.5, 2.0, n).clip(0, 40)
    ports = rng.poisson(2.0, n).clip(0, 8)
    beacon = rng.beta(1.2, 6.0, n)
    suspicious = (rng.random(n) < 0.01).astype(float)
    severity = rng.choice([0.0, 0.25, 0.5, 0.5, 0.75], n)
    criticality = rng.choice([0.25, 0.5, 0.5, 0.75, 1.0], n)
    X = np.column_stack(
        [failed, distinct_users, success_after, off_hours, external, mb_out, ports, beacon,
         suspicious, severity, criticality]
    ).astype(float)
    assert X.shape[1] == len(FEATURE_NAMES)
    return X


def train_model(n: int = 2000, seed: int = 7, n_estimators: int = 150) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators, max_samples=256, random_state=seed, feature_names=FEATURE_NAMES
    )
    return model.fit(generate_baseline(n, seed))


def main(argv: list[str] | None = None) -> Path:
    from sentinelcrew.config import Settings

    parser = argparse.ArgumentParser(description="Train the SentinelCrew triage model")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--trees", type=int, default=150)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or Settings.from_env().model_path
    model = train_model(args.samples, n_estimators=args.trees)
    model.save(out)
    print(f"model saved to {out} ({args.trees} trees, {len(FEATURE_NAMES)} features)")
    return out


if __name__ == "__main__":  # pragma: no cover
    main()
