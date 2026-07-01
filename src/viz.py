"""
Visualization: hero chart (LinkedIn thumbnail) + comparison chart (blog body).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path("output")
OVER_COLOR = "#2ecc71"
UNDER_COLOR = "#e74c3c"
NEUTRAL_COLOR = "#95a5a6"
BG_COLOR = "#ffffff"
TEXT_COLOR = "#2c3e50"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
    }
)


def plot_hero(
    df: pd.DataFrame, path: str | Path = OUTPUT_DIR / "hero_overunder.png"
) -> None:
    """
    Hero chart: horizontal bar of (actual - ELO-xP) per team, sorted.
    Designed as a LinkedIn-ready 1200×630px thumbnail.
    """
    sorted_df = df.sort_values("vs_elo")
    teams = sorted_df.index.tolist()
    deltas = sorted_df["vs_elo"].values
    colors = [OVER_COLOR if d >= 0 else UNDER_COLOR for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    bars = ax.barh(teams, deltas, color=colors, edgecolor="none", height=0.7)
    ax.axvline(0, color=TEXT_COLOR, linewidth=1.0, alpha=0.4)

    # Value labels
    for bar, val in zip(bars, deltas):
        x = bar.get_width()
        ax.text(
            x + (0.05 if x >= 0 else -0.05),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
            color=TEXT_COLOR,
            alpha=0.8,
        )

    over_patch = mpatches.Patch(color=OVER_COLOR, label="Over-performed")
    under_patch = mpatches.Patch(color=UNDER_COLOR, label="Under-performed")
    ax.legend(handles=[over_patch, under_patch], loc="lower right", framealpha=0.0)

    ax.set_xlabel("Actual points − ELO-expected points", fontsize=11)
    ax.set_title(
        "2026 World Cup Group Stage: Who Over- and Under-Performed?",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlim(deltas.min() - 0.8, deltas.max() + 0.8)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_comparison(
    df: pd.DataFrame, path: str | Path = OUTPUT_DIR / "comparison.png"
) -> None:
    """
    Three-bar grouped chart: ELO-xP vs xG-xP vs actual points, sorted by ELO-xP.
    Blog post body chart.
    """
    sorted_df = df.sort_values("elo_xp", ascending=False)
    teams = sorted_df.index.tolist()
    x = np.arange(len(teams))
    # ponytail: xg_xp is all-NaN while FBRef has no xG for this competition; drop that bar
    has_xg = sorted_df["xg_xp"].notna().any()
    width = 0.28 if has_xg else 0.35

    fig, ax = plt.subplots(figsize=(18, 7), dpi=100)
    ax.bar(
        x - width if has_xg else x - width / 2,
        sorted_df["elo_xp"],
        width,
        label="ELO-expected pts",
        color="#3498db",
        alpha=0.85,
    )
    if has_xg:
        ax.bar(
            x,
            sorted_df["xg_xp"],
            width,
            label="xG-expected pts",
            color="#9b59b6",
            alpha=0.85,
        )
    ax.bar(
        x + width if has_xg else x + width / 2,
        sorted_df["actual"],
        width,
        label="Actual pts",
        color="#2ecc71",
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(teams, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Points", fontsize=11)
    ax.set_title(
        "2026 World Cup Group Stage: Expected vs Actual Points",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(framealpha=0.0, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 10)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_scatter(
    df: pd.DataFrame, path: str | Path = OUTPUT_DIR / "scatter.png"
) -> None:
    """
    Scatter: ELO-xP (x) vs actual points (y). Points above diagonal = over-performers.
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    colors = [OVER_COLOR if v >= 0 else UNDER_COLOR for v in df["vs_elo"]]
    ax.scatter(df["elo_xp"], df["actual"], c=colors, s=80, zorder=3)

    # Diagonal
    lim = max(df["elo_xp"].max(), df["actual"].max()) + 0.5
    ax.plot([0, lim], [0, lim], "--", color=TEXT_COLOR, alpha=0.3, linewidth=1)

    for team, row in df.iterrows():
        ax.annotate(
            team,
            (row["elo_xp"], row["actual"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
            alpha=0.85,
        )

    ax.set_xlabel("ELO-expected points (pre-tournament)", fontsize=11)
    ax.set_ylabel("Actual points", fontsize=11)
    ax.set_title(
        "ELO prediction vs reality — 2026 WC Group Stage",
        fontsize=12,
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
