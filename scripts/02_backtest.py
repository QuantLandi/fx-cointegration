"""Single-pair rolling Engle-Granger backtest.

Default cell: GBPUSD / EURUSD, train=63, test=21, z-threshold=3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "data" / "fx_prices.parquet"
OUTPUT_DIR = ROOT / "outputs"

ASSET_1 = "GBPUSD"
ASSET_2 = "EURUSD"
TRAIN_WINDOW = 63
TEST_WINDOW = 21
Z_THRESHOLD = 3.0


def engle_granger_test(y: pd.Series, x: pd.Series) -> bool:
    """True if both series are non-stationary and residuals are stationary (ADF p<=0.05)."""
    if adfuller(y)[1] <= 0.05 or adfuller(x)[1] <= 0.05:
        return False
    resid = sm.OLS(y, sm.add_constant(x)).fit().resid
    return adfuller(resid)[1] <= 0.05


def hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    return float(sm.OLS(y, sm.add_constant(x)).fit().params.iloc[1])


def load_pair(prices: pd.DataFrame, a1: str, a2: str) -> pd.DataFrame:
    """Build pair panel. Keep the first NaN-return row so rolling windows match the notebooks."""
    data = prices[[a1, a2]].copy()
    data[f"log_{a1}"] = np.log(data[a1])
    data[f"log_{a2}"] = np.log(data[a2])
    data[f"log_returns_{a1}"] = np.log(data[a1] / data[a1].shift(1))
    data[f"log_returns_{a2}"] = np.log(data[a2] / data[a2].shift(1))
    data["diff"] = 0.0
    data["z-score"] = 0.0
    data["signal"] = 0
    return data


def run_backtest(
    data: pd.DataFrame,
    a1: str,
    a2: str,
    train_window: int = TRAIN_WINDOW,
    test_window: int = TEST_WINDOW,
    threshold: float = Z_THRESHOLD,
) -> pd.DataFrame:
    """Rolling train/test Engle-Granger pairs strategy (notebook logic)."""
    out = data.copy()
    log1, log2 = f"log_{a1}", f"log_{a2}"
    ret1, ret2 = f"log_returns_{a1}", f"log_returns_{a2}"

    for i in range(train_window, len(out) - test_window + 1, test_window):
        train_start = out.index[i - train_window]
        train_end = out.index[i - 1]
        test_start = out.index[i]
        test_end = out.index[i + test_window - 1]

        train = out.loc[train_start:train_end]
        test = out.loc[test_start:test_end]

        if not engle_granger_test(train[log1], train[log2]):
            continue

        coef = hedge_ratio(train[log1], train[log2])
        train_diff = train[log1] - coef * train[log2]
        out.loc[train_start:train_end, "diff"] = train_diff

        diff_mean = float(train_diff.mean())
        diff_std = float(train_diff.std())
        if diff_std == 0 or np.isnan(diff_std):
            continue

        test_diff = test[log1] - coef * test[log2]
        out.loc[test_start:test_end, "diff"] = test_diff
        z = (test_diff - diff_mean) / diff_std
        out.loc[test_start:test_end, "z-score"] = z

        test_idx = out.loc[test_start:test_end].index
        out.loc[test_idx[z.to_numpy() > threshold], "signal"] = -1
        out.loc[test_idx[z.to_numpy() < -threshold], "signal"] = 1

    out["signal"] = out["signal"].shift(2).fillna(0).astype(int)

    pair = f"{a1}_{a2}"
    out[f"strategy_returns_{pair}"] = (
        out[ret1] * out["signal"] - out[ret2] * out["signal"]
    ).fillna(0.0)
    out[f"strategy_cum_returns_{pair}"] = np.exp(out[f"strategy_returns_{pair}"].cumsum()) - 1

    ann_ret = out[f"strategy_returns_{pair}"].mean() * 252
    ann_vol = out[f"strategy_returns_{pair}"].std() * np.sqrt(252)
    sharpe = float(ann_ret / ann_vol) if ann_vol and ann_vol != 0 else float("nan")
    out[f"Sharpe_ratio_{pair}"] = sharpe
    return out


def main() -> None:
    if not PRICES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PRICES_PATH}. Run: uv run python scripts/01_download_prices.py"
        )

    prices = pd.read_parquet(PRICES_PATH)
    prices.index = pd.to_datetime(prices.index)
    data = load_pair(prices, ASSET_1, ASSET_2)
    result = run_backtest(data, ASSET_1, ASSET_2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        OUTPUT_DIR
        / f"{ASSET_1}_{ASSET_2}_{TRAIN_WINDOW}_{TEST_WINDOW}_z{int(Z_THRESHOLD)}.csv"
    )
    result.to_csv(out_path, index_label="Date")

    pair = f"{ASSET_1}_{ASSET_2}"
    print(
        f"Backtest {ASSET_1}/{ASSET_2} | train={TRAIN_WINDOW} test={TEST_WINDOW} "
        f"z={Z_THRESHOLD}"
    )
    print(f"Rows: {len(result):,}")
    print(f"Signals: {(result['signal'] != 0).sum()} non-zero days")
    print(f"Sharpe: {result[f'Sharpe_ratio_{pair}'].iloc[0]:.6f}")
    print(f"Saved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
