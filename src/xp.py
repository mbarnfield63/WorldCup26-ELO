"""
Expected points computation.

ELO-xP  : pre-tournament ELO win probabilities → expected points per match → summed.
xG-xP   : in-tournament xG via Poisson model → expected points per match → summed.
Actual  : sum of points from actual results.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import poisson

DRAW_RATE = 0.23  # baseline draw probability for international football at even ELO


def _elo_probs(
    home_elo: float, away_elo: float, neutral: bool = True
) -> tuple[float, float, float]:
    """
    Convert ELO ratings to P(home win), P(draw), P(away win).

    Draw probability scales with match closeness:
        p_draw = DRAW_RATE * 4 * W_e * (1 - W_e)
    which peaks at DRAW_RATE for equal teams and shrinks for mismatches.
    """
    elo_diff = home_elo - away_elo  # no home advantage at WC (neutral venues)
    w_e = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_draw = DRAW_RATE * 4.0 * w_e * (1.0 - w_e)
    p_win = w_e - p_draw / 2.0
    p_loss = (1.0 - w_e) - p_draw / 2.0
    return max(0.0, p_win), max(0.0, p_draw), max(0.0, p_loss)


def _xg_probs(
    home_xg: float, away_xg: float, max_goals: int = 10
) -> tuple[float, float, float]:
    """Poisson match simulation from xG values."""
    if math.isnan(home_xg) or math.isnan(away_xg) or home_xg <= 0 or away_xg <= 0:
        return float("nan"), float("nan"), float("nan")

    h_pmf = poisson.pmf(np.arange(max_goals + 1), home_xg)
    a_pmf = poisson.pmf(np.arange(max_goals + 1), away_xg)

    p_win = float(np.sum(np.tril(np.outer(h_pmf, a_pmf), k=-1)))
    p_draw = float(np.sum(np.diag(np.outer(h_pmf, a_pmf))))
    p_loss = 1.0 - p_win - p_draw
    return p_win, p_draw, max(0.0, p_loss)


def _points(result: int) -> int:
    return {1: 3, 0: 1, -1: 0}[result]


def compute_xp(matches: pd.DataFrame, ratings: dict[str, float]) -> pd.DataFrame:
    """
    For each WC 2026 group stage team, compute:
        elo_xp   : sum of ELO-expected points over 3 group games
        xg_xp    : sum of xG-expected points over 3 group games
        actual   : actual points earned

    Returns a DataFrame indexed by team with those three columns.
    """
    rows = []
    for row in matches.itertuples(index=False):
        home, away = row.home_team, row.away_team
        h_elo = ratings.get(home, 1500.0)
        a_elo = ratings.get(away, 1500.0)

        h_win, h_draw, h_loss = _elo_probs(h_elo, a_elo, neutral=True)
        a_win, a_draw, a_loss = h_loss, h_draw, h_win  # symmetric

        h_xg_win, h_xg_draw, h_xg_loss = _xg_probs(row.home_xg, row.away_xg)
        a_xg_win = h_xg_loss
        a_xg_draw = h_xg_draw
        a_xg_loss = h_xg_win

        h_result = (
            1
            if row.home_score > row.away_score
            else (0 if row.home_score == row.away_score else -1)
        )
        a_result = -h_result if h_result != 0 else 0

        rows.append(
            {
                "team": home,
                "elo_xp": 3 * h_win + h_draw,
                "xg_xp": 3 * h_xg_win + h_xg_draw,
                "actual": _points(h_result),
            }
        )
        rows.append(
            {
                "team": away,
                "elo_xp": 3 * a_win + a_draw,
                "xg_xp": 3 * a_xg_win + a_xg_draw,
                "actual": _points(a_result),
            }
        )

    df = pd.DataFrame(rows)
    agg = df.groupby("team").sum(numeric_only=True)

    # Delta columns for the hero chart
    agg["vs_elo"] = agg["actual"] - agg["elo_xp"]
    agg["vs_xg"] = agg["actual"] - agg["xg_xp"]

    return agg.sort_values("vs_elo", ascending=False)
