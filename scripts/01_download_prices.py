"""Download and freeze FX spot prices for the pairs-trading study."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PRICES_PATH = DATA_DIR / "fx_prices.parquet"

TICKERS = (
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "USDCHF=X",
    "USDCAD=X",
    "AUDUSD=X",
    "NZDUSD=X",
)
START = "2007-01-01"
END = "2026-01-01"  # exclusive end → last calendar day of 2025

# Invert so every series is quoted as XXXUSD (matches original notebooks)
INVERT = {
    "USDJPY=X": "JPYUSD",
    "USDCHF=X": "CHFUSD",
    "USDCAD=X": "CADUSD",
}
KEEP = {
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
    "AUDUSD=X": "AUDUSD",
    "NZDUSD=X": "NZDUSD",
}
COLUMNS = ("AUDUSD", "CADUSD", "CHFUSD", "EURUSD", "GBPUSD", "JPYUSD", "NZDUSD")


def _close(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Adj Close" in level0:
            return raw["Adj Close"].copy()
        if "Close" in level0:
            return raw["Close"].copy()
        raise KeyError(f"No Close/Adj Close in columns: {level0.unique().tolist()}")
    if "Adj Close" in raw.columns:
        return raw[["Adj Close"]].copy()
    if "Close" in raw.columns:
        return raw[["Close"]].copy()
    return raw.copy()


def download_prices() -> pd.DataFrame:
    raw = yf.download(
        list(TICKERS),
        start=START,
        end=END,
        auto_adjust=False,
        threads=True,
        progress=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned an empty DataFrame")

    prices = _close(raw)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    out = pd.DataFrame(index=prices.index)
    for yf_name, col in KEEP.items():
        out[col] = prices[yf_name]
    for yf_name, col in INVERT.items():
        out[col] = 1.0 / prices[yf_name]

    out = out.ffill().dropna(how="any")
    return out[list(COLUMNS)]


def load_prices(path: Path = PRICES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: uv run python scripts/01_download_prices.py"
        )
    prices = pd.read_parquet(path)
    prices.index = pd.to_datetime(prices.index)
    return prices


def main() -> None:
    print(f"Downloading FX prices {START} -> {END} ({len(TICKERS)} tickers)...")
    prices = download_prices()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PRICES_PATH)
    print(f"Rows: {len(prices):,} | Columns: {list(prices.columns)}")
    print(f"Range: {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"Saved: {PRICES_PATH.relative_to(ROOT)}")
    print(prices.describe().T[["count", "mean", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
