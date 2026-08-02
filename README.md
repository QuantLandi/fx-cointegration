# FX cointegration pairs trading

Reproducible pipeline for Paper 1 (cointegration-based FX pairs trading).

## Setup

```bash
cd fx-cointegration
uv sync
```

## Pipeline

```bash
uv run python scripts/01_download_prices.py              # freeze data/fx_prices.parquet
uv run python scripts/02_backtest.py --clear-panels       # paper subset (257/21, z=1,2,3)
```

Optional: `--full` for the exploratory multi-window grid.

## Data

Daily Yahoo Finance FX spots, 2007-01-01 to 2024-01-01, seven USD crosses
(AUD, CAD, CHF, EUR, GBP, JPY, NZD), all quoted as XXXUSD.

## Method (sketch)

1. For each directed pair and rolling train/test window, screen log prices with
   Engle–Granger (ADF 5% on residuals; both legs treated as I(1)).
2. If cointegrated, estimate the OLS hedge ratio on the train window and form the
   residual spread; standardize with train mean/SD to get an OOS z-score.
3. Trade when |z| exceeds a threshold (long spread if z < −z*, short if z > z*).
   Signals are lagged two days.
4. Strategy return = signal × (r₁ − r₂); report annualized Sharpe (252).

**Paper subset:** train=257, test=21, z* ∈ {1, 2, 3}, 42 directed pairs (126 configs).

## Outputs

- `outputs/metrics_paper.csv` — one row per config
- `outputs/panels/z_{z}/{leg1}_{leg2}.csv` — e.g. `aud_cad.csv` (USD quote implied)
  (`price_*`, `return_*`, `spread`, `zscore`, `signal`, `strategy_return`, …)  
  For `--full`, filenames also include `_{train}_{test}`.
