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
uv run python scripts/02_backtest.py --clear-panels           # EG (default)
uv run python scripts/02_backtest.py --simple --clear-panels  # no EG screen
uv run python scripts/03_compare_strategies.py               # EG vs simple
```

`--simple` writes `outputs/metrics_simple.csv` and `outputs/panels_simple/`
(same z-rule and PnL; every train window gets an OLS hedge ratio). EG outputs
stay under `metrics_paper.csv` / `panels/`. Compare joins both metrics files
into `outputs/metrics_compare.csv` (ΔSharpe = EG − simple).

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

**Paper subset:** train=257, test=21, z* ∈ {1, 2, 3}, 42 directed pairs (126 configs).

## Outputs

- `outputs/metrics_paper.csv` — EG grid (one row per config)
- `outputs/metrics_simple.csv` — simple grid (`--simple`)
- `outputs/metrics_compare.csv` — joined EG vs simple (`03_compare_strategies.py`)
- `outputs/metrics_*.meta.json` — price freeze + strategy + window/z constants
- `outputs/panels/{leg1}_{leg2}/` — EG panels (e.g. `aud_cad/`)
- `outputs/panels_simple/{leg1}_{leg2}/` — simple panels
  - `prices.csv`, `returns.csv`, `spread.csv`, `zscore.csv`
  - `signal.csv`, `strategy_return.csv`, `strategy_cum_return.csv`
    (z-dependent files use columns `z_1`, `z_2`, `z_3`)

Under EG, `spread` / `zscore` are NaN outside Engle–Granger-pass OOS blocks.
Under `--simple`, almost all OOS blocks are filled (NaN only before the first
window or if train std is zero). Strategy files use 0 for flat days.
