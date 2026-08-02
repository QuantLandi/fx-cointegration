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

`04_portfolio_tables.py` also refreshes Sortino / Calmar / max DD on the pair
metrics CSVs from existing panels (no need to re-run EG just for those columns).

## Data

Daily Yahoo Finance FX spots, 2007-01-01 to 2024-01-01, seven USD crosses
(AUD, CAD, CHF, EUR, GBP, JPY, NZD), all quoted as XXXUSD.

## Method (sketch)

1. For each directed pair and rolling train/test window, screen log prices with
   Engle–Granger: statsmodels `adfuller` defaults (`regression='c'`,
   `autolag='AIC'`), 5% — both legs fail to reject a unit root; OLS residual
   rejects.
2. If cointegrated, reuse that OLS slope as the hedge ratio; form the spread
   `log_1 − β·log_2` and standardize with train mean/SD for an OOS z-score.
3. Trade when |z| exceeds a threshold (long spread if z < −z*, short if z > z*).
   Signals are lagged two days.
4. Strategy return = signal × (r₁ − r₂) (**equal notional**, not β-hedged).
   Annualized Sharpe uses **all calendar days** (flat days as 0), so volatility
   is diluted when often out of market. Metrics `trades` = days with nonzero
   signal, not round-trips.
5. **Paper portfolio:** sum the 42 pair daily returns, divide by 42, then scale
   the EG series so its max drawdown matches the simple strategy
   (`exp(cumsum)−1` wealth, peak-to-trough). Sortino uses the std of strictly
   negative daily returns; Calmar = ann return / |max DD|.

**Paper subset:** train=257, test=21, z* ∈ {1, 2, 3}, 42 directed pairs (126 configs).

## Outputs

```text
outputs/
  coint/
    metrics.csv (+ .meta.json)
    panels/{leg1}_{leg2}/     # prices, returns, spread, zscore, signal, …
  simple/
    metrics.csv (+ .meta.json)
    panels/{leg1}_{leg2}/
  compare/
    metrics.csv               # EG vs simple join (ΔSharpe)
  paper/
    tables/                   # Tables 1–6 CSVs
    portfolio/                # daily / cumulative portfolio series
    figures/                  # fig01 … fig18
```

Under EG, `spread` / `zscore` are NaN outside Engle–Granger-pass OOS blocks.
Under `--simple`, almost all OOS blocks are filled (NaN only before the first
window or if train std is zero). Strategy files use 0 for flat days.

`outputs/` and `local/` are gitignored. Optional local checks (JAE table
tolerances, notebook spot checks) live under `local/`.
