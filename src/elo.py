"""
538-style ELO for international football.

K-factors by match type, goal-margin multiplier (eloratings.net table),
home advantage offset. Bootstraps from 1500 for all teams.
"""

import math
from collections import defaultdict

import pandas as pd

INITIAL_RATING = 1500.0
HOME_ADV = 100.0  # ELO points added to home side; set 0 for neutral venues


def _k_factor(tournament: str) -> float:
    t = tournament.lower()
    if "world cup" in t and "qualif" not in t:
        return 60.0
    if any(
        x in t
        for x in [
            "euro",
            "copa am",
            "africa cup",
            "asian cup",
            "gold cup",
            "nations cup",
            "afcon",
        ]
    ):
        return 50.0
    if "qualif" in t or "nations league" in t or "confederation" in t:
        return 40.0
    if "friendly" in t:
        return 20.0
    return 30.0


def _mov_multiplier(goal_diff: int) -> float:
    """Goal margin multiplier; 1-goal win = 1.0, 2-goal = 1.5, 3+ = (11+N)/8."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8.0


def compute_elo(matches: pd.DataFrame) -> dict[str, float]:
    """
    Run ELO over sorted historical matches. Returns {team_name: elo_rating}.

    Expected columns: date, home_team, away_team, home_score, away_score,
                      tournament, neutral (bool).
    """
    ratings: dict[str, float] = defaultdict(lambda: INITIAL_RATING)

    for row in matches.sort_values("date").itertuples(index=False):
        home, away = row.home_team, row.away_team
        neutral = bool(getattr(row, "neutral", False))

        h_elo, a_elo = ratings[home], ratings[away]
        elo_diff = h_elo - a_elo + (0.0 if neutral else HOME_ADV)
        p_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        h_score, a_score = int(row.home_score), int(row.away_score)
        result = 1.0 if h_score > a_score else 0.5 if h_score == a_score else 0.0

        k = _k_factor(row.tournament)
        delta = k * _mov_multiplier(abs(h_score - a_score)) * (result - p_home)

        ratings[home] += delta
        ratings[away] -= delta

    return dict(ratings)
