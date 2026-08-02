# FX cointegration pairs trading

Reproducible pipeline for Paper 1 (FX cointegration vs simple pairs trading).

```bash
cd fx-cointegration
uv sync
uv run python scripts/01_download_prices.py   # writes data/fx_prices.parquet
uv run python scripts/02_backtest.py          # GBPUSD/EURUSD, 63/21, z=3
```

Frozen prices in `data/` are the source of truth.
