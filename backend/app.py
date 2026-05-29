"""
Flask API + WebSocket replay for live win probability.

Usage (from backend/):
    python app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from src.predict import predict_win_probability

BACKEND_DIR = Path(__file__).resolve().parent
DATASET_PATH = BACKEND_DIR / "data" / "processed" / "win_probability_dataset.csv"
GAMES_INDEX_PATH = BACKEND_DIR / "data" / "raw" / "games_index.csv"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-nba-win-prob")
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

_simulation_stop = False


def _game_labels(game_id: str) -> dict[str, str]:
    """Resolve home/away abbreviations from games_index MATCHUP strings."""
    home_team, away_team, matchup = "HOME", "AWAY", f"Game {game_id}"
    if not GAMES_INDEX_PATH.exists():
        return {"home_team": home_team, "away_team": away_team, "matchup": matchup}

    meta = pd.read_csv(GAMES_INDEX_PATH)
    rows = meta[meta["GAME_ID"].astype(str) == str(game_id)]
    for _, row in rows.iterrows():
        text = str(row.get("MATCHUP", ""))
        if " vs. " in text:
            parts = text.split(" vs. ")
            home_team = parts[0].strip()
            away_team = parts[1].strip()
            matchup = f"{away_team} @ {home_team}"
            break
        if " @ " in text:
            parts = text.split(" @ ")
            away_team = parts[0].strip()
            home_team = parts[1].strip()
            matchup = f"{away_team} @ {home_team}"
            break

    return {"home_team": home_team, "away_team": away_team, "matchup": matchup}


def _load_simulation_game(game_id: str | None = None) -> pd.DataFrame:
    """Load one game's home-perspective timeline for replay."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset missing. Run build_dataset.py after fetch_games."
        )

    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        raise ValueError("Dataset is empty")

    # One row per event: home team perspective only
    home_rows = df[df["home_team"] == 1].copy()
    if game_id:
        home_rows = home_rows[home_rows["game_id"].astype(str) == str(game_id)]
    if home_rows.empty:
        home_rows = df[df["home_team"] == 1].copy()

    gid = home_rows["game_id"].iloc[0]
    game_df = home_rows[home_rows["game_id"] == gid].reset_index(drop=True)
    return game_df


def _run_simulation(game_id: str | None, interval_sec: float = 1.0) -> None:
    """Emit simulated game state over WebSocket."""
    global _simulation_stop
    try:
        game_df = _load_simulation_game(game_id)
    except Exception as exc:  # noqa: BLE001
        socketio.emit("simulation_error", {"error": str(exc)})
        return

    gid = str(game_df["game_id"].iloc[0])
    teams = _game_labels(gid)

    socketio.emit(
        "simulation_started",
        {
            "game_id": gid,
            "events": len(game_df),
            **teams,
            "help": (
                f"Replaying {teams['matchup']}. "
                f"Win % is for {teams['home_team']} (home). "
                f"{teams['away_team']} win % = 100% minus home %."
            ),
        },
    )

    for _, row in game_df.iterrows():
        if _simulation_stop:
            break

        features = {
            "period": int(row["period"]),
            "seconds_remaining": float(row["seconds_remaining"]),
            "score_differential": int(row["score_differential"]),
            "home_team": int(row["home_team"]),
            "possession_team": int(row["possession_team"]),
            "team_fouls": int(row["team_fouls"]),
            "opponent_fouls": int(row["opponent_fouls"]),
        }

        try:
            win_prob = predict_win_probability(features)
        except Exception as exc:  # noqa: BLE001
            win_prob = None
            err = str(exc)
        else:
            err = None

        home_score = int(row.get("home_score", 0))
        away_score = int(row.get("away_score", 0))
        payload = {
            "game_id": str(row["game_id"]),
            "period": int(row["period"]),
            "seconds_remaining": float(row["seconds_remaining"]),
            "score_differential": int(row["score_differential"]),
            "home_score": home_score,
            "away_score": away_score,
            "home_team": teams["home_team"],
            "away_team": teams["away_team"],
            "matchup": teams["matchup"],
            "win_probability": win_prob,
            "away_win_probability": (1.0 - win_prob) if win_prob is not None else None,
            "leading_team": teams["home_team"]
            if home_score > away_score
            else teams["away_team"]
            if away_score > home_score
            else "Tied",
            "actual_win_label": int(row.get("win_label", 0)),
            "error": err,
        }
        socketio.emit("game_update", payload)
        socketio.sleep(interval_sec)

    if not _simulation_stop:
        socketio.emit("simulation_complete", {"game_id": str(game_df["game_id"].iloc[0])})


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    model_ok = (BACKEND_DIR / "models" / "win_probability_model.pt").exists()
    data_ok = DATASET_PATH.exists()
    return jsonify(
        {
            "status": "ok",
            "service": "nba-live-win-probability",
            "model_loaded": model_ok,
            "dataset_available": data_ok,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    body = request.get_json(silent=True) or {}
    required = ("period", "seconds_remaining", "score_differential")
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    features = {
        "period": body.get("period", 1),
        "seconds_remaining": body.get("seconds_remaining", 0),
        "score_differential": body.get("score_differential", 0),
        "home_team": body.get("home_team", 1),
        "possession_team": body.get("possession_team", 0),
        "team_fouls": body.get("team_fouls", 0),
        "opponent_fouls": body.get("opponent_fouls", 0),
    }

    try:
        prob = predict_win_probability(features)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify({"win_probability": prob, "features": features})


@socketio.on("connect")
def on_connect():
    emit("connected", {"message": "Connected to NBA win probability simulation"})


@socketio.on("disconnect")
def on_disconnect():
    global _simulation_stop
    _simulation_stop = True


@socketio.on("start_simulation")
def start_simulation(data):
    global _simulation_stop

    _simulation_stop = True
    socketio.sleep(0.1)
    _simulation_stop = False

    game_id = (data or {}).get("game_id")
    interval = float((data or {}).get("interval_sec", 1.0))
    interval = max(0.2, min(interval, 5.0))

    socketio.start_background_task(_run_simulation, game_id, interval)
    emit("simulation_ack", {"game_id": game_id, "interval_sec": interval})


@socketio.on("stop_simulation")
def stop_simulation():
    global _simulation_stop
    _simulation_stop = True
    emit("simulation_stopped", {})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://127.0.0.1:{port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
