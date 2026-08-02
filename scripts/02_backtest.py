"""Rolling Engle-Granger FX pairs backtest (Paper 1 subset).

42 directed pairs, train=257, test=21, z in {1, 2, 3}.

  uv run python scripts/02_backtest.py
  uv run python scripts/02_backtest.py --clear-panels
  uv run python scripts/02_backtest.py --simple --clear-panels

Panel layout (per pair):
  outputs/panels/aud_cad/            # EG (default)
  outputs/panels_simple/aud_cad/     # --simple
    prices.csv, returns.csv, spread.csv, zscore.csv
    signal.csv, strategy_return.csv, strategy_cum_return.csv
  (z-dependent files use columns z_1, z_2, z_3)
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from datetime import datetime, timezone
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
PANELS_SIMPLE_DIR = OUTPUT_DIR / "panels_simple"

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


def engle_granger_hedge(y: pd.Series, x: pd.Series) -> float | None:
    """Engle–Granger screen at 5%; return OLS hedge ratio or None.

    Uses statsmodels ``adfuller`` defaults: regression='c', autolag='AIC'
    (max lag chosen by AIC). Both legs must fail to reject a unit root
    (p > 0.05); the OLS residual must reject (p <= 0.05). The returned
    coefficient is the slope on x from OLS of y on (const, x).
    """
    if adfuller(y)[1] <= 0.05 or adfuller(x)[1] <= 0.05:
        return None
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    if adfuller(fit.resid)[1] > 0.05:
        return None
    return float(fit.params.iloc[1])


def ols_hedge(y: pd.Series, x: pd.Series) -> float:
    """OLS hedge ratio (slope on x) with no cointegration screen."""
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
    *,
    require_coint: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rolling OOS spread/z-score on train/test blocks.

    If ``require_coint`` (default), skip windows that fail Engle–Granger.
    If False (simple pairs), always use the OLS hedge ratio.

    Days outside an active OOS block keep NaN spread/zscore (not traded).
    Returns (panel, window_stats).
    """
    out = data.copy()
    spread = pd.Series(np.nan, index=out.index, dtype=float)
    zscore = pd.Series(np.nan, index=out.index, dtype=float)
    stats = {"windows": 0, "passed": 0, "skipped_eg": 0, "skipped_std": 0}

    for i in range(train_window, len(out) - test_window + 1, test_window):
        train = out.iloc[i - train_window : i]
        test = out.iloc[i : i + test_window]
        stats["windows"] += 1

        if require_coint:
            coef = engle_granger_hedge(train["log_1"], train["log_2"])
            if coef is None:
                stats["skipped_eg"] += 1
                continue
        else:
            coef = ols_hedge(train["log_1"], train["log_2"])

        train_spread = train["log_1"] - coef * train["log_2"]
        std = float(train_spread.std())
        if std == 0.0 or np.isnan(std):
            stats["skipped_std"] += 1
            continue

        mean = float(train_spread.mean())
        test_spread = test["log_1"] - coef * test["log_2"]
        spread.loc[test.index] = test_spread
        zscore.loc[test.index] = (test_spread - mean) / std
        stats["passed"] += 1

    out["spread"] = spread
    out["zscore"] = zscore
    return out, stats


def apply_threshold(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Z-score rule with a 2-day lag. NaN z (inactive window) -> flat.

    PnL is equal-notional: signal * (r1 - r2). The EG hedge ratio is used
    only to form the spread/z-score, not to size the second leg.
    """
    out = data.copy()
    z = out["zscore"].to_numpy(dtype=float)
    # NaN comparisons are False, so inactive days stay flat (same as |z| < threshold).
    signal = np.select(
        [z > threshold, z < -threshold],
        [-1, 1],
        default=0,
    ).astype(int)
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
    """Summary metrics for one config.

    Sharpe uses all calendar days (flat days as 0 return), so vol is diluted
    when the strategy is often out of the market. ``trades`` counts days with
    nonzero signal, not round-trips.
    """
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


def z_col(threshold: float) -> str:
    """Column name for a z-threshold (z_1, z_2, z_1.5, …)."""
    return f"z_{threshold:g}"


def pair_dir(a1: str, a2: str, panels_root: Path) -> Path:
    path = panels_root / f"{leg_code(a1)}_{leg_code(a2)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pair_panels(
    scored: pd.DataFrame,
    by_z: dict[float, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Write one-concept-per-file panels; z-dependent series as columns z_1, z_2, …"""
    scored[["price_1", "price_2"]].to_csv(out_dir / "prices.csv", index_label="date")
    scored[["return_1", "return_2"]].to_csv(out_dir / "returns.csv", index_label="date")
    scored[["spread"]].to_csv(out_dir / "spread.csv", index_label="date")
    scored[["zscore"]].to_csv(out_dir / "zscore.csv", index_label="date")

    signal = pd.DataFrame({z_col(z): df["signal"] for z, df in by_z.items()})
    strategy_return = pd.DataFrame(
        {z_col(z): df["strategy_return"] for z, df in by_z.items()}
    )
    strategy_cum_return = pd.DataFrame(
        {z_col(z): df["strategy_cum_return"] for z, df in by_z.items()}
    )
    signal.to_csv(out_dir / "signal.csv", index_label="date")
    strategy_return.to_csv(out_dir / "strategy_return.csv", index_label="date")
    strategy_cum_return.to_csv(out_dir / "strategy_cum_return.csv", index_label="date")


def run_grid(
    prices: pd.DataFrame,
    *,
    require_coint: bool,
    panels_root: Path,
) -> pd.DataFrame:
    train_window, test_window = PAPER_WINDOW
    pairs = directed_pairs()
    rows: list[dict] = []
    totals = {"windows": 0, "passed": 0, "skipped_eg": 0, "skipped_std": 0}

    for done, (a1, a2) in enumerate(pairs, start=1):
        print(
            f"[{done}/{len(pairs)}] {a1}/{a2} "
            f"train={train_window} test={test_window}",
            flush=True,
        )
        scored, stats = compute_zscores(
            load_pair(prices, a1, a2),
            train_window,
            test_window,
            require_coint=require_coint,
        )
        for key in totals:
            totals[key] += stats[key]
        by_z: dict[float, pd.DataFrame] = {}
        for z in Z_THRESHOLDS:
            result = apply_threshold(scored, z)
            by_z[z] = result
            rows.append(summarize(result, a1, a2, train_window, test_window, z))
        save_pair_panels(scored, by_z, pair_dir(a1, a2, panels_root))

    if require_coint:
        print(
            "Window screen: "
            f"{totals['passed']}/{totals['windows']} passed EG, "
            f"{totals['skipped_eg']} failed EG, "
            f"{totals['skipped_std']} skipped (zero/NaN train std)",
            flush=True,
        )
    else:
        print(
            "Window screen (simple, no EG gate): "
            f"{totals['passed']}/{totals['windows']} active, "
            f"{totals['skipped_std']} skipped (zero/NaN train std)",
            flush=True,
        )
    return pd.DataFrame(rows)


def write_metrics_provenance(
    prices: pd.DataFrame,
    metrics_path: Path,
    *,
    strategy: str,
) -> Path:
    """Sidecar JSON recording which price freeze and constants produced metrics."""
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": strategy,
        "metrics_file": metrics_path.relative_to(ROOT).as_posix(),
        "prices_file": PRICES_PATH.relative_to(ROOT).as_posix(),
        "prices_mtime_utc": datetime.fromtimestamp(
            PRICES_PATH.stat().st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prices_rows": int(len(prices)),
        "prices_start": str(prices.index.min().date()),
        "prices_end": str(prices.index.max().date()),
        "currencies": list(CURRENCIES),
        "train_window": PAPER_WINDOW[0],
        "test_window": PAPER_WINDOW[1],
        "z_thresholds": list(Z_THRESHOLDS),
        "n_pairs": len(directed_pairs()),
        "n_configs": len(directed_pairs()) * len(Z_THRESHOLDS),
        "signal_lag_days": 2,
        "pnl": "equal_notional_r1_minus_r2",
    }
    meta_path = metrics_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Simple pairs (no EG screen). Writes metrics_simple / panels_simple.",
    )
    parser.add_argument(
        "--clear-panels",
        action="store_true",
        help="Delete the panels directory for this mode before writing.",
    )
    args = parser.parse_args()

    require_coint = not args.simple
    panels_root = PANELS_DIR if require_coint else PANELS_SIMPLE_DIR
    strategy = "engle_granger" if require_coint else "simple_no_eg_screen"
    metrics_name = "metrics_paper.csv" if require_coint else "metrics_simple.csv"

    if not PRICES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PRICES_PATH}. Run: uv run python scripts/01_download_prices.py"
        )

    if args.clear_panels and panels_root.exists():
        shutil.rmtree(panels_root)
        print(f"Cleared {panels_root.relative_to(ROOT)}")

    prices = pd.read_parquet(PRICES_PATH)
    prices.index = pd.to_datetime(prices.index)

    n_pairs = len(directed_pairs())
    n_configs = n_pairs * len(Z_THRESHOLDS)
    label = "EG" if require_coint else "simple"
    print(
        f"Running {label} grid: {n_pairs} pairs x "
        f"window {PAPER_WINDOW[0]}/{PAPER_WINDOW[1]} x "
        f"{len(Z_THRESHOLDS)} z = {n_configs} configs"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # OLS/ADF can emit CollinearityWarning / RuntimeWarning on singular windows;
    # those cases are counted via skipped_eg / skipped_std in run_grid.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=CollinearityWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        metrics = run_grid(
            prices, require_coint=require_coint, panels_root=panels_root
        )

    out_path = OUTPUT_DIR / metrics_name
    metrics.to_csv(out_path, index=False)
    meta_path = write_metrics_provenance(prices, out_path, strategy=strategy)
    print(f"Saved metrics: {out_path.relative_to(ROOT)} ({len(metrics)} rows)")
    print(f"Saved provenance: {meta_path.relative_to(ROOT)}")
    print(f"Saved panels:  {n_pairs} pair folders under {panels_root.relative_to(ROOT)}/")
    print(metrics.sort_values("sharpe", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
