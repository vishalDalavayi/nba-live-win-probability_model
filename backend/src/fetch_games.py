"""
Fetch a small sample of NBA playoff games and save play-by-play logs as CSV.

Uses PlayByPlayV3 (V2 is deprecated and returns empty responses).

Usage (from backend/):
    python -m src.fetch_games
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3
from nba_api.stats.static import teams as nba_teams

BACKEND_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BACKEND_DIR / "data" / "raw"

DEFAULT_SEASON = "2023-24"
MAX_GAMES = 8
REQUEST_DELAY_SEC = 0.6


def _season_playoff_games(season: str = DEFAULT_SEASON) -> pd.DataFrame:
    """Return playoff game rows from LeagueGameFinder."""
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Playoffs",
        league_id_nullable="00",
    )
    games = finder.get_data_frames()[0]
    if games.empty:
        raise RuntimeError(f"No playoff games found for season {season}")
    return games


def _fetch_play_by_play(game_id: str) -> pd.DataFrame:
    """Fetch play-by-play for a single game with retries (V3 API)."""
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
            df = pbp.get_data_frames()[0]
            if df is not None and not df.empty:
                df["GAME_ID"] = game_id
                return df
            last_err = RuntimeError("empty play-by-play response")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(REQUEST_DELAY_SEC * (attempt + 1))
    raise RuntimeError(f"Failed to fetch play-by-play for {game_id}: {last_err}")


def fetch_and_save(season: str = DEFAULT_SEASON, max_games: int = MAX_GAMES) -> list[Path]:
    """Download playoff PBP and write CSV files to data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_games = _season_playoff_games(season=season)
    game_list = all_games.drop_duplicates(subset=["GAME_ID"]).head(max_games)
    saved: list[Path] = []

    print(f"Fetching up to {len(game_list)} playoff games for {season}...")
    for _, row in game_list.iterrows():
        game_id = str(row["GAME_ID"])
        out_path = RAW_DIR / f"pbp_{game_id}.csv"
        if out_path.exists():
            print(f"  Skip (exists): {out_path.name}")
            saved.append(out_path)
            continue

        try:
            pbp = _fetch_play_by_play(game_id)
            pbp.to_csv(out_path, index=False)
            saved.append(out_path)
            matchup = f"{row.get('MATCHUP', game_id)}"
            print(f"  Saved {out_path.name} ({len(pbp)} events) — {matchup}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: skipped {game_id}: {exc}")

        time.sleep(REQUEST_DELAY_SEC)

    game_ids = game_list["GAME_ID"].tolist()
    meta = all_games[all_games["GAME_ID"].isin(game_ids)][
        ["GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ID"]
    ].copy()
    meta_path = RAW_DIR / "games_index.csv"
    meta.to_csv(meta_path, index=False)
    print(f"Wrote index: {meta_path} ({len(saved)} PBP files)")
    return saved


if __name__ == "__main__":
    print(f"nba_api teams loaded: {len(nba_teams.get_teams())}")
    paths = fetch_and_save()
    print(f"Done. {len(paths)} play-by-play files in {RAW_DIR}")
