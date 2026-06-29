"""
WC26 ELO pipeline — run end-to-end.

Usage:
    uv run python main.py

Requires a Kaggle API token (~/.kaggle/kaggle.json).
"""

from pathlib import Path

import kagglehub

from src.data import load_historical, scrape_wc26_matches
from src.elo import compute_elo
from src.xp import compute_xp
from src.viz import plot_hero, plot_comparison, plot_scatter

KAGGLE_DATASET = "martj42/international-football-results-from-1872-to-2017"


def _get_csv() -> Path:
    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv = dataset_dir / "results.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"results.csv not found in downloaded dataset at {dataset_dir}"
        )
    return csv


def main() -> None:
    print("=== 1. Loading historical results ===")
    historical = load_historical(_get_csv())
    print(f"    {len(historical):,} matches loaded (up to 2026-06-10)")

    print("=== 2. Computing ELO ratings ===")
    ratings = compute_elo(historical)
    print(f"    {len(ratings)} teams rated")

    # Sanity check: top 10 entering the tournament
    top10 = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:10]
    print("    Top 10 ELO entering WC 2026:")
    for rank, (team, elo) in enumerate(top10, 1):
        print(f"      {rank:2}. {team:<30} {elo:.0f}")

    print("=== 3. Scraping WC 2026 group stage results + xG ===")
    wc26 = scrape_wc26_matches()
    print(f"    {len(wc26)} group stage matches loaded")

    print("=== 4. Computing expected points ===")
    xp_df = compute_xp(wc26, ratings)
    print(xp_df[["elo_xp", "xg_xp", "actual", "vs_elo", "vs_xg"]].to_string())

    print("=== 5. Generating charts ===")
    plot_hero(xp_df)
    plot_comparison(xp_df)
    plot_scatter(xp_df)
    print("Done. Charts saved to output/")


if __name__ == "__main__":
    main()
