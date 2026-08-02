"""Rolling Engle-Granger FX pairs backtest (Paper 1 subset).

Default: 42 directed pairs, train=257, test=21, z in {1, 2, 3}.

  uv run python scripts/02_backtest.py
  uv run python scripts/02_backtest.py --clear-panels
  uv run python scripts/02_backtest.py --full   # exploratory window grid
"""

from __future__ import annotations

import argparse
import shutil
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import CollinearityWarning

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "data" / "fx_prices.parquet"
OUTPUT_DIR = ROOT / "outputs"
PANELS_DIR = OUTPUT_DIR / "panels"

CURRENCIES = (
    "AUDUSD",
    "CADUSD",
    "CHFUSD",
    "EURUSD",
    "GBPUSD",
    "JPYUSD",
    "NZDUSD",
)

PAPER_WINDOW = (257, 21)
Z_THRESHOLDS = (1.0, 2.0, 3.0)

# Used only with --full
WINDOWS_FULL = (
    (63, 1),
    (63, 5),
    (63, 21),
    (128, 1),
    (128, 5),
    (128, 21),
    (128, 63),
    (257, 1),
    (257, 5),
    (257, 21),
    (257, 63),
    (257, 128),
)

PANEL_COLUMNS = (
    "price_1",
    "price_2",
    "return_1",
    "return_2",
    "spread",
    "zscore",
    "signal",
    "strategy_return",
    "strategy_cum_return",
)


def engle_granger_test(y: pd.Series, x: pd.Series) -> bool:
    """ADF 5%: both legs non-stationary, OLS residual stationary."""
    if adfuller(y)[1] <= 0.05 or adfuller(x)[1] <= 0.05:
        return False
    resid = sm.OLS(y, sm.add_constant(x)).fit().resid
    return bool(adfuller(resid)[1] <= 0.05)


def hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    return float(sm.OLS(y, sm.add_constant(x)).fit().params.iloc[1])


def load_pair(prices: pd.DataFrame, a1: str, a2: str) -> pd.DataFrame:
    """Levels + log returns. Keep the leading NaN return for window alignment."""
    price_1 = prices[a1]
    price_2 = prices[a2]
    log_1 = np.log(price_1)
    log_2 = np.log(price_2)
    return pd.DataFrame(
        {
            "price_1": price_1,
            "price_2": price_2,
            "return_1": log_1.diff(),
            "return_2": log_2.diff(),
            "log_1": log_1,
            "log_2": log_2,
        },
        index=prices.index,
    )


def compute_zscores(
    data: pd.DataFrame,
    train_window: int,
    test_window: int,
) -> pd.DataFrame:
    """Rolling EG screen; write OOS spread/z-score on test windows that pass."""
    out = data.copy()
    spread = pd.Series(0.0, index=out.index)
    zscore = pd.Series(0.0, index=out.index)

    for i in range(train_window, len(out) - test_window + 1, test_window):
        train = out.iloc[i - train_window : i]
        test = out.iloc[i : i + test_window]

        if not engle_granger_test(train["log_1"], train["log_2"]):
            continue

        coef = hedge_ratio(train["log_1"], train["log_2"])
        train_spread = train["log_1"] - coef * train["log_2"]
        std = float(train_spread.std())
        if std == 0.0 or np.isnan(std):
            continue

        mean = float(train_spread.mean())
        test_spread = test["log_1"] - coef * test["log_2"]
        spread.loc[test.index] = test_spread
        zscore.loc[test.index] = (test_spread - mean) / std

    out["spread"] = spread
    out["zscore"] = zscore
    return out


def apply_threshold(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Z-score rule with a 2-day lag."""
    out = data.copy()
    z = out["zscore"].to_numpy()
    signal = np.zeros(len(out), dtype=int)
    signal[z > threshold] = -1
    signal[z < -threshold] = 1
    out["signal"] = pd.Series(signal, index=out.index).shift(2).fillna(0).astype(int)
    out["strategy_return"] = (
        out["signal"] * (out["return_1"] - out["return_2"])
    ).fillna(0.0)
    out["strategy_cum_return"] = np.exp(out["strategy_return"].cumsum()) - 1.0
    return out


def summarize(
    result: pd.DataFrame,
    a1: str,
    a2: str,
    train_window: int,
    test_window: int,
    threshold: float,
) -> dict:
    rets = result["strategy_return"]
    ann_ret = float(rets.mean() * 252)
    ann_vol = float(rets.std() * np.sqrt(252))
    sharpe = float(ann_ret / ann_vol) if ann_vol else float("nan")
    return {
        "currency_1": a1,
        "currency_2": a2,
        "train_window": train_window,
        "test_window": test_window,
        "z_threshold": threshold,
        "sharpe": sharpe,
        "ann_return": ann_ret,
        "ann_volatility": ann_vol,
        "trades": int((result["signal"] != 0).sum()),
    }


def directed_pairs() -> list[tuple[str, str]]:
    return list(permutations(CURRENCIES, 2))


def leg_code(symbol: str) -> str:
    """AUDUSD -> aud (USD quote is implicit)."""
    if not symbol.endswith("USD"):
        raise ValueError(f"Expected XXXUSD symbol, got {symbol}")
    return symbol[:-3].lower()


def panel_path(
    a1: str,
    a2: str,
    train_window: int,
    test_window: int,
    z: float,
    *,
    include_window_in_name: bool,
) -> Path:
    folder = PANELS_DIR / f"z_{int(z)}"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{leg_code(a1)}_{leg_code(a2)}"
    if include_window_in_name:
        stem = f"{stem}_{train_window}_{test_window}"
    return folder / f"{stem}.csv"


def run_grid(
    prices: pd.DataFrame,
    windows: list[tuple[int, int]],
    thresholds: list[float],
) -> pd.DataFrame:
    pairs = directed_pairs()
    rows: list[dict] = []
    total = len(pairs) * len(windows)
    include_window_in_name = len(windows) > 1

    for done, ((a1, a2), (train_window, test_window)) in enumerate(
        ((pair, window) for pair in pairs for window in windows),
        start=1,
    ):
        print(
            f"[{done}/{total}] {a1}/{a2} train={train_window} test={test_window}",
            flush=True,
        )
        scored = compute_zscores(load_pair(prices, a1, a2), train_window, test_window)
        for z in thresholds:
            result = apply_threshold(scored, z)
            rows.append(summarize(result, a1, a2, train_window, test_window, z))
            result.loc[:, PANEL_COLUMNS].to_csv(
                panel_path(
                    a1,
                    a2,
                    train_window,
                    test_window,
                    z,
                    include_window_in_name=include_window_in_name,
                ),
                index_label="date",
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Exploratory grid over all notebook windows (default is paper 257/21).",
    )
    parser.add_argument(
        "--clear-panels",
        action="store_true",
        help="Delete outputs/panels before writing.",
    )
    args = parser.parse_args()

    if args.full:
        label, windows, out_name = "full", list(WINDOWS_FULL), "metrics_full.csv"
    else:
        label, windows, out_name = "paper", [PAPER_WINDOW], "metrics_paper.csv"
    thresholds = list(Z_THRESHOLDS)

    if not PRICES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PRICES_PATH}. Run: uv run python scripts/01_download_prices.py"
        )

    if args.clear_panels and PANELS_DIR.exists():
        shutil.rmtree(PANELS_DIR)
        print(f"Cleared {PANELS_DIR.relative_to(ROOT)}")

    prices = pd.read_parquet(PRICES_PATH)
    prices.index = pd.to_datetime(prices.index)

    n_configs = len(directed_pairs()) * len(windows) * len(thresholds)
    print(
        f"Running {label}: {len(directed_pairs())} pairs x "
        f"{len(windows)} windows x {len(thresholds)} z = {n_configs} configs"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=CollinearityWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        metrics = run_grid(prices, windows, thresholds)

    out_path = OUTPUT_DIR / out_name
    metrics.to_csv(out_path, index=False)
    print(f"Saved metrics: {out_path.relative_to(ROOT)} ({len(metrics)} rows)")
    print(f"Saved panels:  {n_configs} files under {PANELS_DIR.relative_to(ROOT)}/")
    print(metrics.sort_values("sharpe", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
