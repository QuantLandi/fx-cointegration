"""Rolling Engle-Granger FX pairs backtest (undirected pairs).

21 unordered pairs, train=257, test=21, z in {1, 2, 3}.
Default: Engle–Granger screen. --simple: always-trade (no EG gate).

  uv run python scripts/02_backtest.py --clear-panels
  uv run python scripts/02_backtest.py --simple --clear-panels

Panel layout (per pair, alphabetical legs):
  outputs/coint/panels/aud_cad/
  outputs/simple/panels/aud_cad/
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import CollinearityWarning

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "data" / "fx_prices.parquet"
OUTPUT_DIR = ROOT / "outputs"
COINT_DIR = OUTPUT_DIR / "coint"
SIMPLE_DIR = OUTPUT_DIR / "simple"

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
# Round-trip bp of pair notional; charge (κ/1e4)*|Δs|/2 per day.
COST_BPS = (0.0, 1.0, 2.0, 5.0)
BASELINE_COST_BP = 2.0


def engle_granger_detail(
    y: pd.Series, x: pd.Series
) -> tuple[bool, float | None, float | None]:
    """EG screen at 5%.

    Returns (passed, slope_on_x, residual_adf_stat).
    ADF defaults: regression='c', autolag='AIC'. Both legs must fail to reject
    a unit root (p > 0.05); residual must reject (p <= 0.05).
    """
    if adfuller(y)[1] <= 0.05 or adfuller(x)[1] <= 0.05:
        return False, None, None
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    adf_stat, adf_p, *_ = adfuller(fit.resid)
    if adf_p > 0.05:
        return False, None, None
    return True, float(fit.params.iloc[1]), float(adf_stat)


def ols_hedge(y: pd.Series, x: pd.Series) -> float:
    """OLS hedge ratio (slope on x) with no cointegration screen."""
    return float(sm.OLS(y, sm.add_constant(x)).fit().params.iloc[1])


def alpha_hedge_from_reverse(beta_rev: float) -> float | None:
    """Map log_2 = a + beta_rev log_1 to alphabetical hedge in log_1 - γ log_2."""
    if beta_rev == 0.0 or np.isnan(beta_rev):
        return None
    return 1.0 / beta_rev


def choose_hedge(
    log_1: pd.Series,
    log_2: pd.Series,
    *,
    require_coint: bool,
) -> float | None:
    """Alphabetical hedge γ for spread log_1 - γ log_2, or None if flat.

    Legs are already alphabetical. If ``require_coint``, run Engle–Granger on
    both OLS orientations and keep the passing residual with the more negative
    ADF statistic (map reverse orientation into the alphabetical spread). If
    ``require_coint`` is False, always use OLS of log_1 on log_2.
    """
    if not require_coint:
        return ols_hedge(log_1, log_2)

    candidates: list[tuple[float, float]] = []
    pass_fwd, beta_fwd, stat_fwd = engle_granger_detail(log_1, log_2)
    pass_rev, beta_rev, stat_rev = engle_granger_detail(log_2, log_1)
    if pass_fwd and beta_fwd is not None and stat_fwd is not None:
        candidates.append((stat_fwd, beta_fwd))
    if pass_rev and beta_rev is not None and stat_rev is not None:
        gamma = alpha_hedge_from_reverse(beta_rev)
        if gamma is not None:
            candidates.append((stat_rev, gamma))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


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
    require_coint: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rolling OOS spread/z-score on train/test blocks."""
    out = data.copy()
    spread = pd.Series(np.nan, index=out.index, dtype=float)
    zscore = pd.Series(np.nan, index=out.index, dtype=float)
    stats = {"windows": 0, "passed": 0, "skipped_eg": 0, "skipped_std": 0}

    for i in range(train_window, len(out) - test_window + 1, test_window):
        train = out.iloc[i - train_window : i]
        test = out.iloc[i : i + test_window]
        stats["windows"] += 1

        coef = choose_hedge(
            train["log_1"], train["log_2"], require_coint=require_coint
        )
        if coef is None:
            stats["skipped_eg"] += 1
            continue

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

    Panel ``strategy_return`` is *gross* (zero transaction cost). Net returns
    for a cost grid are applied later via ``apply_transaction_costs``.
    """
    out = data.copy()
    z = out["zscore"].to_numpy(dtype=float)
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


def apply_transaction_costs(
    gross: pd.Series,
    signal: pd.Series,
    cost_rt_bp: float,
) -> pd.Series:
    """Subtract round-trip costs charged on |Δsignal|/2.

    ``cost_rt_bp`` is basis points of pair notional for a full open→close
    round trip (κ). Opening or closing (|Δs|=1) costs κ/2 bp; a flip (|Δs|=2)
    costs a full κ bp.
    """
    gross = gross.fillna(0.0)
    if cost_rt_bp == 0.0:
        return gross
    sig = signal.fillna(0).astype(float)
    turnover = sig.diff().abs()
    if len(turnover):
        turnover.iloc[0] = float(abs(sig.iloc[0]))
    cost = (cost_rt_bp / 1e4) * (turnover / 2.0)
    return gross - cost.fillna(0.0)


def metrics_from_returns(rets: pd.Series) -> dict[str, float]:
    rets = rets.fillna(0.0)
    ann_ret = float(rets.mean() * 252)
    ann_vol = float(rets.std() * np.sqrt(252))
    sharpe = float(ann_ret / ann_vol) if ann_vol else float("nan")
    cum = np.exp(rets.cumsum()) - 1.0
    drawdown = cum - cum.cummax()
    max_dd = float(drawdown.min()) if len(drawdown) else float("nan")
    downside = rets[rets < 0.0]
    downside_std = float(downside.std()) if len(downside) else float("nan")
    sortino = (
        float(ann_ret / (downside_std * np.sqrt(252)))
        if downside_std and not np.isnan(downside_std)
        else float("nan")
    )
    calmar = (
        float(ann_ret / abs(max_dd))
        if max_dd and not np.isnan(max_dd)
        else float("nan")
    )
    return {
        "ann_return": ann_ret,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
    }


def summarize(
    result: pd.DataFrame,
    a1: str,
    a2: str,
    train_window: int,
    test_window: int,
    threshold: float,
    *,
    cost_bp: float = 0.0,
) -> dict:
    rets = apply_transaction_costs(
        result["strategy_return"], result["signal"], cost_bp
    )
    m = metrics_from_returns(rets)
    return {
        "currency_1": a1,
        "currency_2": a2,
        "train_window": train_window,
        "test_window": test_window,
        "z_threshold": threshold,
        "cost_bp": cost_bp,
        **m,
        "trades": int((result["signal"] != 0).sum()),
    }


def undirected_pairs() -> list[tuple[str, str]]:
    return [(a, b) if a < b else (b, a) for a, b in combinations(CURRENCIES, 2)]


def leg_code(symbol: str) -> str:
    if not symbol.endswith("USD"):
        raise ValueError(f"Expected XXXUSD symbol, got {symbol}")
    return symbol[:-3].lower()


def z_col(threshold: float) -> str:
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
    pairs = undirected_pairs()
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
            for cost_bp in COST_BPS:
                rows.append(
                    summarize(
                        result,
                        a1,
                        a2,
                        train_window,
                        test_window,
                        z,
                        cost_bp=cost_bp,
                    )
                )
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
        "cost_bps": list(COST_BPS),
        "baseline_cost_bp": BASELINE_COST_BP,
        "cost_model": "rt_bp_on_abs_dsignal_over_2",
        "n_pairs": len(undirected_pairs()),
        "n_configs": (
            len(undirected_pairs()) * len(Z_THRESHOLDS) * len(COST_BPS)
        ),
        "pair_universe": "undirected_alphabetical",
        "signal_lag_days": 2,
        "pnl": "equal_notional_r1_minus_r2",
        "panels_strategy_return": "gross_zero_cost",
    }
    meta_path = metrics_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Simple pairs (no EG screen). Writes outputs/simple/.",
    )
    parser.add_argument(
        "--clear-panels",
        action="store_true",
        help="Delete the panels directory for this mode before writing.",
    )
    args = parser.parse_args()

    require_coint = not args.simple
    strategy_root = COINT_DIR if require_coint else SIMPLE_DIR
    panels_root = strategy_root / "panels"
    out_path = strategy_root / "metrics.csv"
    strategy = "engle_granger" if require_coint else "simple_no_eg_screen"

    if not PRICES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PRICES_PATH}. Run: uv run python scripts/01_download_prices.py"
        )

    if args.clear_panels and panels_root.exists():
        shutil.rmtree(panels_root)
        print(f"Cleared {panels_root.relative_to(ROOT)}")

    prices = pd.read_parquet(PRICES_PATH)
    prices.index = pd.to_datetime(prices.index)

    n_pairs = len(undirected_pairs())
    n_configs = n_pairs * len(Z_THRESHOLDS) * len(COST_BPS)
    label = "EG" if require_coint else "simple"
    print(
        f"Running {label} grid: {n_pairs} undirected pairs x "
        f"window {PAPER_WINDOW[0]}/{PAPER_WINDOW[1]} x "
        f"{len(Z_THRESHOLDS)} z x {len(COST_BPS)} cost = {n_configs} configs"
    )

    strategy_root.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=CollinearityWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        metrics = run_grid(
            prices, require_coint=require_coint, panels_root=panels_root
        )

    metrics.to_csv(out_path, index=False)
    meta_path = write_metrics_provenance(prices, out_path, strategy=strategy)
    print(f"Saved metrics: {out_path.relative_to(ROOT)} ({len(metrics)} rows)")
    print(f"Saved provenance: {meta_path.relative_to(ROOT)}")
    print(f"Saved panels:  {n_pairs} pair folders under {panels_root.relative_to(ROOT)}/")
    print(metrics.sort_values("sharpe", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
