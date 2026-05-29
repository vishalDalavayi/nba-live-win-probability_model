"""
Build training dataset from raw play-by-play CSV files (V2 or V3 schema).

Usage (from backend/):
    python -m src.build_dataset
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BACKEND_DIR / "data" / "raw"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "win_probability_dataset.csv"

REGULATION_PERIOD_SEC = 12 * 60
OT_PERIOD_SEC = 5 * 60


def parse_clock_to_seconds(clock_str: str) -> float:
    """Parse clock strings: '11:45', 'PT12M00.00S', etc."""
    if pd.isna(clock_str) or not str(clock_str).strip():
        return 0.0
    text = str(clock_str).strip().upper()

    iso = re.match(r"PT(\d+)M([\d.]+)S", text)
    if iso:
        return int(iso.group(1)) * 60 + float(iso.group(2))

    if ":" in text:
        parts = text.split(":")
        try:
            return int(parts[0]) * 60 + float(parts[1])
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def seconds_remaining_in_game(period: int, clock_sec: float, max_period: int = 4) -> float:
    period = int(period) if pd.notna(period) else 1
    if period <= 4:
        remaining_periods = max(0, 4 - period)
        return remaining_periods * REGULATION_PERIOD_SEC + clock_sec
    ot_index = period - 4
    remaining_ot = max(0, max_period - period) if max_period > 4 else 0
    _ = ot_index  # reserved for future OT logic
    return remaining_ot * OT_PERIOD_SEC + clock_sec


def parse_score(score_val) -> tuple[int, int]:
    if pd.isna(score_val):
        return 0, 0
    text = str(score_val).strip()
    nums = re.findall(r"\d+", text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    if len(nums) == 1:
        return int(nums[0]), 0
    return 0, 0


def _safe_int(val, default: int = 0) -> int:
    try:
        if pd.isna(val):
            return default
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _is_v3_schema(pbp: pd.DataFrame) -> bool:
    return "scoreHome" in pbp.columns or "gameId" in pbp.columns


def _resolve_team_ids_v3(pbp: pd.DataFrame) -> tuple[int | None, int | None]:
    """Infer home/away team IDs from location flags (h/v)."""
    home_id, away_id = None, None
    if "location" in pbp.columns and "teamId" in pbp.columns:
        h = pbp[pbp["location"].astype(str).str.lower() == "h"]["teamId"]
        v = pbp[pbp["location"].astype(str).str.lower() == "v"]["teamId"]
        h = h[h.astype(float) > 0]
        v = v[v.astype(float) > 0]
        if not h.empty:
            home_id = _safe_int(h.mode().iloc[0], 0) or None
        if not v.empty:
            away_id = _safe_int(v.mode().iloc[0], 0) or None

    if home_id and away_id:
        return home_id, away_id

    team_ids = []
    if "teamId" in pbp.columns:
        team_ids = [
            _safe_int(t)
            for t in pbp["teamId"].dropna().unique()
            if _safe_int(t) > 0
        ]
    if len(team_ids) >= 2:
        return team_ids[0], team_ids[1]
    if len(team_ids) == 1:
        return team_ids[0], team_ids[0]
    return None, None


def infer_possession_v2(row: pd.Series, home_team_id: int, away_team_id: int) -> int | None:
    for col in ("PLAYER1_TEAM_ID", "PLAYER2_TEAM_ID", "PLAYER3_TEAM_ID"):
        if col in row.index and pd.notna(row[col]):
            try:
                tid = int(row[col])
                if tid in (home_team_id, away_team_id):
                    return tid
            except (ValueError, TypeError):
                continue
    return None


def process_game_file(path: Path, game_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    pbp = pd.read_csv(path)
    if pbp.empty:
        return pd.DataFrame()

    v3 = _is_v3_schema(pbp)
    if "GAME_ID" in pbp.columns:
        game_id = str(pbp["GAME_ID"].iloc[0])
    elif "gameId" in pbp.columns:
        game_id = str(pbp["gameId"].iloc[0])
    else:
        game_id = path.stem.replace("pbp_", "")

    if v3:
        home_id, away_id = _resolve_team_ids_v3(pbp)
        period_col = "period"
        clock_col = "clock"
        home_score_col, away_score_col = "scoreHome", "scoreAway"
    else:
        home_id, away_id = None, None
        period_col, clock_col = "PERIOD", "PCTIMESTRING"
        home_score_col, away_score_col = None, None

    if game_meta is not None and not game_meta.empty:
        g_rows = game_meta[game_meta["GAME_ID"].astype(str) == game_id]
        if len(g_rows) >= 2 and (home_id is None or away_id is None):
            home_id = _safe_int(g_rows.iloc[0].get("TEAM_ID"))
            away_id = _safe_int(g_rows.iloc[1].get("TEAM_ID"))

    if home_id is None or away_id is None:
        home_id, away_id = _resolve_team_ids_v3(pbp) if v3 else (None, None)

    if home_id is None or away_id is None:
        print(f"  Warning: could not resolve team IDs for {game_id}, skipping")
        return pd.DataFrame()

    # Final score
    if v3:
        home_score_final = _safe_int(pbp[home_score_col].dropna().iloc[-1] if home_score_col in pbp else 0)
        away_score_final = _safe_int(pbp[away_score_col].dropna().iloc[-1] if away_score_col in pbp else 0)
    else:
        home_score_final, away_score_final = 0, 0
        if "SCORE" in pbp.columns:
            for val in reversed(pbp["SCORE"].tolist()):
                if pd.notna(val) and str(val).strip():
                    home_score_final, away_score_final = parse_score(val)
                    break

    home_won = 1 if home_score_final > away_score_final else 0
    away_won = 1 - home_won if home_score_final != away_score_final else 0

    max_period = int(pbp[period_col].max()) if period_col in pbp.columns else 4

    rows: list[dict] = []
    home_fouls, away_fouls = 0, 0
    h_score, a_score = 0, 0

    for _, event in pbp.iterrows():
        period = _safe_int(event.get(period_col), 1)
        clock_sec = parse_clock_to_seconds(event.get(clock_col, "0:00"))
        sec_remaining = seconds_remaining_in_game(period, clock_sec, max_period=max_period)

        if v3:
            h_score = _safe_int(event.get(home_score_col), h_score)
            a_score = _safe_int(event.get(away_score_col), a_score)
            tid = _safe_int(event.get("teamId"), 0)
            possession = tid if tid in (home_id, away_id) else None
            desc = str(event.get("description", "") or "").lower()
            action = str(event.get("actionType", "") or "").lower()
        else:
            if "SCORE" in event.index and pd.notna(event["SCORE"]):
                h_score, a_score = parse_score(event["SCORE"])
            possession = infer_possession_v2(event, home_id, away_id)
            desc = " ".join(
                str(event.get(c, "") or "")
                for c in ("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION")
            ).lower()
            action = desc

        if "foul" in desc or "foul" in action:
            if possession == home_id:
                home_fouls += 1
            elif possession == away_id:
                away_fouls += 1

        for team_id, is_home, t_score, o_score, team_f, opp_f, won in (
            (home_id, 1, h_score, a_score, home_fouls, away_fouls, home_won),
            (away_id, 0, a_score, h_score, away_fouls, home_fouls, away_won),
        ):
            rows.append(
                {
                    "game_id": game_id,
                    "team_id": team_id,
                    "period": period,
                    "seconds_remaining": sec_remaining,
                    "score_differential": t_score - o_score,
                    "home_team": is_home,
                    "possession_team": 1 if possession == team_id else 0,
                    "team_fouls": team_f,
                    "opponent_fouls": opp_f,
                    "win_label": won,
                    "home_score": h_score,
                    "away_score": a_score,
                }
            )

    return pd.DataFrame(rows)


def build_dataset() -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(RAW_DIR.glob("pbp_*.csv"))
    if not raw_files:
        raise FileNotFoundError(
            f"No raw PBP files in {RAW_DIR}. Run: python -m src.fetch_games"
        )

    meta_path = RAW_DIR / "games_index.csv"
    game_meta = pd.read_csv(meta_path) if meta_path.exists() else None

    frames = []
    for path in raw_files:
        print(f"Processing {path.name}...")
        df = process_game_file(path, game_meta=game_meta)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No training rows produced. Check raw data files.")

    dataset = pd.concat(frames, ignore_index=True)
    dataset.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(dataset)} rows -> {OUTPUT_CSV}")
    return OUTPUT_CSV


if __name__ == "__main__":
    build_dataset()
