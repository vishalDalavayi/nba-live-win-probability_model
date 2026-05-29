"""
Load trained model and predict win probability from game-state features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.model import WinProbabilityNet

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BACKEND_DIR / "models" / "win_probability_model.pt"
SCALER_PATH = BACKEND_DIR / "models" / "scaler.json"

_model: WinProbabilityNet | None = None
_scaler: dict | None = None
_feature_columns: list[str] | None = None


def _load_artifacts() -> None:
    global _model, _scaler, _feature_columns

    if _model is not None:
        return

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Model or scaler not found. Train first: python -m src.train_model"
        )

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    _feature_columns = checkpoint["feature_columns"]
    _model = WinProbabilityNet(input_dim=checkpoint["input_dim"])
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    _scaler = json.loads(SCALER_PATH.read_text())


def predict_win_probability(features: dict) -> float:
    """
    Predict win probability for the team described by `features`.

    Expected keys (subset of training features):
        period, seconds_remaining, score_differential, home_team,
        possession_team, team_fouls, opponent_fouls
    """
    _load_artifacts()
    assert _scaler is not None and _feature_columns is not None and _model is not None

    mean = np.array(_scaler["mean"], dtype=np.float32)
    scale = np.array(_scaler["scale"], dtype=np.float32)

    row = []
    for col in _feature_columns:
        val = features.get(col, 0)
        try:
            row.append(float(val))
        except (TypeError, ValueError):
            row.append(0.0)

    x = np.array(row, dtype=np.float32)
    x = (x - mean) / np.where(scale == 0, 1.0, scale)
    tensor_x = torch.tensor(x.reshape(1, -1), dtype=torch.float32)

    with torch.no_grad():
        prob = float(_model(tensor_x).item())

    return float(np.clip(prob, 0.0, 1.0))


def features_from_row(row: dict, perspective: str = "home") -> dict:
    """Build model feature dict from a processed dataset row."""
    if perspective == "home":
        return {
            "period": row.get("period", 1),
            "seconds_remaining": row.get("seconds_remaining", 0),
            "score_differential": row.get("score_differential", 0),
            "home_team": 1,
            "possession_team": row.get("possession_team", 0),
            "team_fouls": row.get("team_fouls", 0),
            "opponent_fouls": row.get("opponent_fouls", 0),
        }
    # away perspective: flip differential and home flag
    home_diff = row.get("score_differential", 0)
    return {
        "period": row.get("period", 1),
        "seconds_remaining": row.get("seconds_remaining", 0),
        "score_differential": -home_diff if row.get("home_team", 1) == 1 else home_diff,
        "home_team": 0,
        "possession_team": 1 - int(row.get("possession_team", 0)),
        "team_fouls": row.get("opponent_fouls", 0),
        "opponent_fouls": row.get("team_fouls", 0),
    }
