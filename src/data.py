"""
Data loading: Kaggle historical results CSV + FBRef WC 2026 group stage via soccerdata.
"""

from pathlib import Path

import pandas as pd
import soccerdata as sd

# WC 2026 kicked off 2026-06-11; exclude from ELO training window
WC26_START = pd.Timestamp("2026-06-11")

# FBRef sometimes uses different country names than the Kaggle CSV.
# Map FBRef → Kaggle to keep ELO keys consistent.
_TEAM_NAME_MAP: dict[str, str] = {
    "United States": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "North Macedonia": "North Macedonia",
    "Trinidad and Tobago": "Trinidad and Tobago",
    "Cape Verde": "Cape Verde Islands",
}


def _normalise_team(name: str) -> str:
    return _TEAM_NAME_MAP.get(name, name)


def load_historical(path: str | Path) -> pd.DataFrame:
    """
    Load the Kaggle international results CSV, drop rows with missing scores,
    and exclude any matches on or after WC26_START.

    Download from: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
    Save to data/raw/results.csv
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["date"] < WC26_START].copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


def scrape_wc26_matches() -> pd.DataFrame:
    """
    Fetch WC 2026 group stage schedule (results + xG) from FBRef via soccerdata.

    Returns a DataFrame with columns:
        date, home_team, away_team, home_score, away_score, home_xg, away_xg, round
    """
    fbref = sd.FBref(leagues="INT-World Cup", seasons=2026)
    raw = fbref.read_schedule()

    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index()

    raw.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in raw.columns]

    # FBRef returns a combined "H–A" score string; split it
    if "score" in raw.columns and "home_score" not in raw.columns:
        parsed = (
            raw["score"]
            .str.replace("−", "-")
            .str.replace("–", "-")
            .str.split("-", n=1, expand=True)
        )
        raw["home_score"] = pd.to_numeric(parsed[0].str.strip(), errors="coerce")
        raw["away_score"] = pd.to_numeric(parsed[1].str.strip(), errors="coerce")

    # xG is not in the schedule; try shot events and aggregate per game
    # ponytail: falls back to NaN so xg_xp is just skipped in viz
    raw["home_xg"] = float("nan")
    raw["away_xg"] = float("nan")
    try:
        shots = fbref.read_shot_events()
        if isinstance(shots.index, pd.MultiIndex):
            shots = shots.reset_index()
        shots.columns = [
            c.lower().replace(" ", "_").replace("-", "_") for c in shots.columns
        ]
        xg_col = next((c for c in shots.columns if "xg" in c and "psxg" not in c), None)
        if xg_col and "game_id" in shots.columns and "team" in shots.columns:
            home_map = raw.set_index("game_id")["home_team"].to_dict()
            shots["is_home"] = shots["game_id"].map(home_map) == shots["team"]
            xg_agg = (
                shots.groupby(["game_id", "is_home"])[xg_col]
                .sum()
                .unstack(fill_value=0)
            )
            if True in xg_agg.columns:
                raw["home_xg"] = raw["game_id"].map(xg_agg[True])
            if False in xg_agg.columns:
                raw["away_xg"] = raw["game_id"].map(xg_agg[False])
    except Exception:
        pass

    # Filter to completed group stage matches
    round_col = next(
        (c for c in ["round", "stage", "round_name"] if c in raw.columns), None
    )
    if round_col is None:
        raise ValueError(f"No round column found. Available: {list(raw.columns)}")

    group_mask = raw[round_col].str.lower().str.contains("group", na=False)
    played_mask = raw["home_score"].notna() & raw["away_score"].notna()
    df = raw[group_mask & played_mask].copy()

    if df.empty:
        raise ValueError(
            f"No completed group stage matches found.\n"
            f"Round values seen: {raw[round_col].unique().tolist()}\n"
            f"Played rows: {played_mask.sum()} of {len(raw)}"
        )

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    for col in ("home_team", "away_team"):
        if col in df.columns:
            df[col] = df[col].map(_normalise_team).fillna(df[col])

    return (
        df[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "home_xg",
                "away_xg",
                round_col,
            ]
        ]
        .rename(columns={round_col: "round"})
        .reset_index(drop=True)
    )
