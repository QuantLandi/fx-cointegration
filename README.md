# FX cointegration pairs trading

Reproducible pipeline for Paper 1 (FX cointegration vs simple pairs trading).

```bash
cd fx-cointegration
uv sync
uv run python download_prices.py   # writes data/fx_prices.parquet
```

Frozen prices in `data/` are the source of truth; re-download only to refresh.
