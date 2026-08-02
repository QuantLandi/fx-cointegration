# FX cointegration pairs trading

Replication code for Lemishko, Landi & Caicedo-Llano (2024),
*Cointegration-Based Strategies in Forex Pairs Trading*
([SSRN](https://ssrn.com/abstract=4771108)).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```bash
git clone https://github.com/QuantLandi/fx-cointegration.git
cd fx-cointegration
uv sync
```

Frozen prices are in `data/fx_prices.parquet`. Re-download only if you need a new freeze:

```bash
uv run python scripts/01_download_prices.py
```

## Pipeline

From the repo root:

```bash
uv run python scripts/02_backtest.py --clear-panels           # EG → outputs/coint/
uv run python scripts/02_backtest.py --simple --clear-panels  # → outputs/simple/
uv run python scripts/03_compare_strategies.py               # → outputs/compare/
uv run python scripts/04_portfolio_tables.py                 # → outputs/paper/tables|portfolio
uv run python scripts/05_plot_figures.py                     # → outputs/paper/figures/
```

`04_portfolio_tables.py` also refreshes pair `metrics.csv` from existing panels
(including the `{0,1,2,5}` bp cost grid), so you need not re-run EG just for
metrics/portfolio tables.

## Data

Daily Yahoo Finance FX spots, 2007-01-01 to 2025-12-31, seven USD crosses
(AUD, CAD, CHF, EUR, GBP, JPY, NZD), all quoted as XXXUSD.

## Method (sketch)

1. For each **undirected** pair (21 = C(7,2), alphabetical legs) and rolling
   train/test window, screen log prices with Engle–Granger at 5%: both OLS
   orientations are tested; among passes, keep the clearer residual ADF
   (more negative t-stat) and map the hedge into `log_1 − β·log_2`.
2. If cointegrated, form that spread and standardize with train mean/SD for an
   OOS z-score.
3. Trade when |z| exceeds a threshold (long spread if z < −z*, short if z > z*).
   Signals are lagged two days.
4. Strategy return = signal × (r₁ − r₂) (**equal notional**, not β-hedged).
   Annualized Sharpe uses **all calendar days** (flat days as 0), so volatility
   is diluted when often out of market. Metrics `trades` = days with nonzero
   signal, not round-trips.
5. **Transaction costs:** round-trip κ bp of pair notional charged as
   `(κ/1e4)·|Δsignal|/2` (open/close = κ/2 each; flip = κ). Grid
   `{0, 1, 2, 5}` bp; headline **κ = 2**. Panels store gross returns; costs are
   applied when building metrics and portfolio tables.
6. **Paper portfolio:** sum the 21 pair daily returns, divide by 21. Main tables
   are **unlevered**. A companion table scales each strategy ex-post to **10%
   annualized vol** so return/MDD levels are comparable (Sharpe unchanged).
   Cumulative-return figures scale the EG path to **equal ex-post daily vol** vs
   simple (visuals only). Sortino uses the std of strictly negative daily returns;
   Calmar = ann return / |max DD|.

**Paper subset:** train=257, test=21, z* ∈ {1, 2, 3}, 21 undirected pairs,
cost κ ∈ {0, 1, 2, 5} bp (baseline 2). Simple benchmark: `--simple` (no EG gate).

## Outputs

```text
outputs/
  coint/
    metrics.csv (+ .meta.json)   # rows × z × cost_bp; panels are gross
    panels/{leg1}_{leg2}/        # prices, returns, spread, zscore, signal, …
  simple/
    metrics.csv (+ .meta.json)
    panels/{leg1}_{leg2}/
  compare/
    metrics.csv                  # EG vs simple join (ΔSharpe), keyed by cost_bp
  paper/
    tables/                      # unlevered / target-vol / cost-sensitivity CSVs
    portfolio/                   # daily / cumulative series at baseline κ=2
    figures/                     # fig01–fig05
```

Under EG, `spread` / `zscore` are NaN outside Engle–Granger-pass OOS blocks.
Under `--simple`, almost all OOS blocks are filled (NaN only before the first
window or if train std is zero). Strategy return panels are **gross** (zero cost);
flat days are 0. Net returns apply `(κ/1e4)·|Δsignal|/2` in metrics and `04`.

`outputs/` and `local/` are gitignored. Optional local checks (JAE table
tolerances, notebook spot checks) live under `local/`.

Typst draft: `paper/main.typ` (figures via `paper/figures` → `outputs/paper/figures`).
