# WC26 ELO — Who Over- and Under-Performed at the 2026 World Cup?

A data pipeline that builds national team ELO ratings from 150 years of international football, then uses them to quantify which teams exceeded or fell short of expectations in the 2026 World Cup group stage.

---

## The Idea

Raw results don't tell you much on their own — beating a weak team is expected. To know whether a result is *surprising*, you need a baseline.

This project builds that baseline in two ways:

**ELO-expected points** — bootstrapped 538-style ELO ratings from every international match since 1872. Each team's rating entering the tournament captures their quality across qualifying, friendlies, and previous tournaments. From those ratings, I compute the expected points each team *should* earn in each group game.

**xG-expected points** — even within the tournament, results can be lucky or unlucky. I use FBRef match-level xG (expected goals) and a Poisson model to compute what the shot quality alone would have predicted, independent of the actual scoreline.

Comparing these three numbers — ELO-xP, xG-xP, actual points — tells a layered story:

| Pattern | Reading |
|---|---|
| actual > ELO-xP > xG-xP | Over-performed both model and chance — genuinely surprised |
| actual > ELO-xP, actual ≈ xG-xP | Result was deserved by shot quality, but better than expected given squad strength |
| actual < xG-xP < ELO-xP | Underperformed and were unlucky — double trouble |
| xG-xP > actual > ELO-xP | Created enough to win but couldn't convert — better than it looked on paper |

---

## Method

### ELO model

- **Data:** Kaggle "International Football Results 1872–2026" (~50k matches)
- **Starting rating:** 1500 for all teams (ratings converge after ~20 matches)
- **K-factors by match type:**
  - World Cup: 60
  - Continental championships (Euros, Copa América, AFCON, etc.): 50
  - Qualifiers / Nations League: 40
  - Friendlies: 20
- **Goal margin multiplier:** 1.0 (≤1 goal), 1.5 (2 goals), (11+N)/8 for N>2 goals
- **Home advantage:** +100 ELO points (zero for neutral venues)
- **Win probability:** standard ELO formula `1 / (1 + 10^(-Δ/400))`

### Draw probability

Draw rate is not constant — it peaks for evenly matched teams and falls for mismatches:

```
p_draw = 0.23 × 4 × W_e × (1 − W_e)
```

where W_e is the ELO-expected score. This keeps all three probabilities non-negative even at extreme ELO differences.

### xG → expected points

For each match, home and away xG values from FBRef are treated as Poisson rates. P(win), P(draw), P(loss) are computed by summing over the joint goal-count distribution up to 10 goals per side.

---

## Output

Three charts saved to `output/`:

| File | Purpose |
|---|---|
| `hero_overunder.png` | Hero bar chart — over/under-performance ranked by ELO gap. LinkedIn-ready. |
| `comparison.png` | Grouped bars: ELO-xP vs xG-xP vs actual, sorted by pre-tournament expectation. |
| `scatter.png` | Scatter: ELO-xP (x) vs actual points (y). Points above diagonal = over-performers. |

---

## Running It

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
# Install dependencies
uv sync

# Download the Kaggle dataset and save to:
# data/raw/results.csv
# https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

# Run the full pipeline
uv run python main.py
```

---

## Stack

| Layer | Tool |
|---|---|
| Historical data | Kaggle CSV (international football results 1872–2026) |
| WC 2026 match data + xG | FBRef via `soccerdata` |
| ELO computation | Pure Python + pandas |
| xG model | `scipy` Poisson |
| Visualisation | `matplotlib` + `seaborn` |

---

## Future Extensions

- Monte Carlo group stage simulation (pre-tournament, for bracket prediction)
- Extend analysis through knockout rounds as the tournament progresses
- Confederation-level breakdown of over/under-performance patterns
