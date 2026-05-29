"""
Train win probability model on processed dataset.

Usage (from backend/):
    python -m src.train_model
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.model import WinProbabilityNet

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BACKEND_DIR / "data" / "processed" / "win_probability_dataset.csv"
MODEL_PATH = BACKEND_DIR / "models" / "win_probability_model.pt"
SCALER_PATH = BACKEND_DIR / "models" / "scaler.json"

FEATURE_COLUMNS = [
    "period",
    "seconds_remaining",
    "score_differential",
    "home_team",
    "possession_team",
    "team_fouls",
    "opponent_fouls",
]

EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run: python -m src.build_dataset"
        )

    df = pd.read_csv(DATA_PATH)
    for col in FEATURE_COLUMNS + ["win_label"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    X = df[FEATURE_COLUMNS].astype(float).values
    y = df["win_label"].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    BACKEND_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
    scaler_payload = {
        "feature_columns": FEATURE_COLUMNS,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    SCALER_PATH.write_text(json.dumps(scaler_payload, indent=2))

    train_tensor_x = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    test_tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32)

    model = WinProbabilityNet(input_dim=len(FEATURE_COLUMNS))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    n = len(train_tensor_x)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            batch_x = train_tensor_x[idx]
            batch_y = train_tensor_y[idx]

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} — loss: {epoch_loss / max(1, n // BATCH_SIZE):.4f}")

    model.eval()
    with torch.no_grad():
        test_probs = model(test_tensor_x).numpy().flatten()

    test_preds_binary = (test_probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, test_preds_binary)
    ll = log_loss(y_test, np.clip(test_probs, 1e-6, 1 - 1e-6))
    brier = brier_score_loss(y_test, test_probs)

    print("\n--- Test metrics ---")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Log loss:     {ll:.4f}")
    print(f"Brier score:  {brier:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(FEATURE_COLUMNS),
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    print(f"\nModel saved -> {MODEL_PATH}")
    print(f"Scaler saved -> {SCALER_PATH}")


if __name__ == "__main__":
    train()
