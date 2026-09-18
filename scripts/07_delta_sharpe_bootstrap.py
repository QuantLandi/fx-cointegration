"""Inference on the EG vs always-trade portfolio horse race (unlevered 1/21 book).

Tests whether the Sharpe, Sortino and mean-return advantages of the
Engle-Granger book over the always-trade book are distinguishable from zero.

All designs use a paired circular block bootstrap: blocks of dates are drawn
once and applied to BOTH books, so contemporaneous cross-book correlation is
preserved. Sharpe and mean return are additionally studentized by a HAC
standard error (Ledoit-Wolf 2008) and cross-checked against the HAC delta
method (Jobson-Korkie / Memmel with a Newey-West long-run covariance).

Sortino here follows the repo definition (annualized return over the std of
strictly negative daily returns), which has no closed-form HAC standard error,
so it is reported with percentile intervals only.

Reads existing panels only; no 02 re-run.

  uv run python scripts/07_delta_sharpe_bootstrap.py
  uv run python scripts/07_delta_sharpe_bootstrap.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "outputs" / "paper" / "tables"
OUT_CSV = TABLES_DIR / "tables_delta_sharpe_bootstrap.csv"

ANN = 252.0
Z_THRESHOLDS = (1.0, 2.0, 3.0)
BASELINE_COST_BP = 2.0
COST_ROBUSTNESS_BP = (0.0, 1.0, 5.0)
PRIMARY_BLOCK = 21  # test-window length
BLOCK_SENSITIVITY = (5, 21, 63)
N_BOOT = 10_000
SEED = 20260918


def _load_module(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def books(pt, bt, z: float, cost_bp: float) -> tuple[pd.Series, pd.Series]:
    """Aligned daily net returns of the two unlevered 1/21 books."""
    eg = pt.load_portfolio_returns(
        pt.PANELS_EG,
        z,
        cost_rt_bp=cost_bp,
        apply_transaction_costs=bt.apply_transaction_costs,
    )
    simple = pt.load_portfolio_returns(
        pt.PANELS_SIMPLE,
        z,
        cost_rt_bp=cost_bp,
        apply_transaction_costs=bt.apply_transaction_costs,
    )
    idx = eg.index.union(simple.index).sort_values()
    return eg.reindex(idx).fillna(0.0), simple.reindex(idx).fillna(0.0)


def sharpe_ann(x: np.ndarray) -> float:
    """Annualized Sharpe, ddof=1, matching metrics_from_returns in 02."""
    sd = float(np.std(x, ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(np.sqrt(ANN) * np.mean(x) / sd)


def sortino_ann(x: np.ndarray) -> float:
    """Annualized Sortino, matching metrics_from_returns in 02.

    Denominator is the ddof=1 standard deviation of strictly negative daily
    returns (the repo convention), not a below-target downside deviation.
    """
    downside = x[x < 0.0]
    if len(downside) < 2:
        return float("nan")
    sd = float(np.std(downside, ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(np.sqrt(ANN) * np.mean(x) / sd)


def nw_bandwidth(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_cov(v: np.ndarray, bandwidth: int) -> np.ndarray:
    """Newey-West (Bartlett) long-run covariance of demeaned columns of v."""
    n = v.shape[0]
    s = v.T @ v / n
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        g = v[lag:].T @ v[:-lag] / n
        s = s + weight * (g + g.T)
    return s


def estimates(
    r1: np.ndarray,
    r2: np.ndarray,
    r1sq: np.ndarray,
    r2sq: np.ndarray,
    bandwidth: int,
) -> tuple[float, float, float, float]:
    """(dSharpe, dMean, HAC SE of dSharpe, HAC SE of dMean), all annualized.

    One Newey-West long-run covariance of (r1, r2, r1^2, r2^2) serves both:
    the Sharpe difference via the Ledoit-Wolf gradient, and the mean
    difference via Var(r1 - r2) = S00 + S11 - 2 S01.
    """
    n = len(r1)
    mu1, mu2 = float(np.mean(r1)), float(np.mean(r2))
    g1, g2 = float(np.mean(r1sq)), float(np.mean(r2sq))
    var1, var2 = g1 - mu1**2, g2 - mu2**2
    nan = float("nan")
    if var1 <= 0.0 or var2 <= 0.0:
        return nan, nan, nan, nan

    # Point estimates use ddof=1, matching metrics_from_returns in 02.
    scale = n / (n - 1.0)
    d_sharpe = float(
        np.sqrt(ANN) * (mu1 / np.sqrt(var1 * scale) - mu2 / np.sqrt(var2 * scale))
    )
    d_mean = float(ANN * (mu1 - mu2))

    v = np.empty((n, 4))
    v[:, 0] = r1 - mu1
    v[:, 1] = r2 - mu2
    v[:, 2] = r1sq - g1
    v[:, 3] = r2sq - g2
    s = hac_cov(v, bandwidth)

    grad = np.array(
        [
            g1 / var1**1.5,
            -g2 / var2**1.5,
            -mu1 / (2.0 * var1**1.5),
            mu2 / (2.0 * var2**1.5),
        ]
    )
    var_sharpe = float(grad @ s @ grad) / n
    lrv_diff = float(s[0, 0] + s[1, 1] - 2.0 * s[0, 1])
    se_sharpe = float(np.sqrt(ANN * var_sharpe)) if var_sharpe > 0.0 else nan
    se_mean = float(ANN * np.sqrt(lrv_diff / n)) if lrv_diff > 0.0 else nan
    return d_sharpe, d_mean, se_sharpe, se_mean


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap index draw."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block)
    idx = (starts[:, None] + offsets[None, :]).ravel() % n
    return idx[:n]


def normal_two_sided_p(t: float) -> float:
    from math import erfc, sqrt

    if not np.isfinite(t):
        return float("nan")
    return float(erfc(abs(t) / sqrt(2.0)))


def bootstrap_cell(
    r1: np.ndarray,
    r2: np.ndarray,
    *,
    block: int,
    n_boot: int,
    bandwidth: int,
    seed: int,
) -> list[dict]:
    """Studentized + percentile bootstrap for dSharpe and dMean on one design."""
    n = len(r1)
    r1sq, r2sq = r1**2, r2**2
    d_sharpe, d_mean, se_sharpe, se_mean = estimates(r1, r2, r1sq, r2sq, bandwidth)
    d_sortino = sortino_ann(r1) - sortino_ann(r2)

    rng = np.random.default_rng(seed)
    boot_sharpe = np.empty(n_boot)
    boot_mean = np.empty(n_boot)
    boot_sortino = np.empty(n_boot)
    t_sharpe = np.empty(n_boot)
    t_mean = np.empty(n_boot)
    for b in range(n_boot):
        idx = block_indices(n, block, rng)
        b1, b2 = r1[idx], r2[idx]
        ds, dm, s_s, s_m = estimates(b1, b2, r1sq[idx], r2sq[idx], bandwidth)
        boot_sharpe[b] = ds
        boot_mean[b] = dm
        boot_sortino[b] = sortino_ann(b1) - sortino_ann(b2)
        t_sharpe[b] = (ds - d_sharpe) / s_s if s_s and np.isfinite(s_s) else np.nan
        t_mean[b] = (dm - d_mean) / s_m if s_m and np.isfinite(s_m) else np.nan

    nan = float("nan")
    rows = []
    for stat, point, se, boot, tstar, eg_val, simple_val in (
        (
            "delta_sharpe",
            d_sharpe,
            se_sharpe,
            boot_sharpe,
            t_sharpe,
            sharpe_ann(r1),
            sharpe_ann(r2),
        ),
        (
            "delta_sortino",
            d_sortino,
            nan,
            boot_sortino,
            None,
            sortino_ann(r1),
            sortino_ann(r2),
        ),
        (
            "delta_ann_return_pct",
            d_mean * 100.0,
            se_mean * 100.0,
            boot_mean * 100.0,
            t_mean,
            float(ANN * np.mean(r1) * 100.0),
            float(ANN * np.mean(r2) * 100.0),
        ),
    ):
        t_obs = point / se if se and np.isfinite(se) else nan
        if tstar is None:
            p_stud, ci_t = nan, (nan, nan)
        else:
            finite = tstar[np.isfinite(tstar)]
            if len(finite) and np.isfinite(t_obs):
                p_stud = (1.0 + np.sum(np.abs(finite) >= abs(t_obs))) / (
                    len(finite) + 1.0
                )
                q = float(np.quantile(np.abs(finite), 0.95))
                ci_t = (point - q * se, point + q * se)
            else:
                p_stud, ci_t = nan, (nan, nan)

        good = boot[np.isfinite(boot)]
        if len(good):
            # Two-sided percentile p-value: how far 0 sits in the resampled tail.
            # (1 + count) / (B + 1) keeps the floor at 2/(B+1) rather than zero.
            denom = len(good) + 1.0
            tail = min(
                (1.0 + float(np.sum(good <= 0.0))) / denom,
                (1.0 + float(np.sum(good >= 0.0))) / denom,
            )
            p_pct = min(1.0, 2.0 * tail)
            ci_p = (
                float(np.quantile(good, 0.025)),
                float(np.quantile(good, 0.975)),
            )
            boot_sd = float(np.std(good, ddof=1))
        else:
            p_pct, ci_p, boot_sd = nan, (nan, nan), nan

        rows.append(
            {
                "stat": stat,
                "eg": eg_val,
                "simple": simple_val,
                "point": point,
                "hac_se": se,
                "boot_sd": boot_sd,
                "t_stat": t_obs,
                "p_hac_normal": normal_two_sided_p(t_obs),
                "p_boot_studentized": p_stud,
                "p_boot_percentile": p_pct,
                "ci95_studentized_lo": ci_t[0],
                "ci95_studentized_hi": ci_t[1],
                "ci95_percentile_lo": ci_p[0],
                "ci95_percentile_hi": ci_p[1],
                "n_obs": n,
                "block": block,
                "hac_bandwidth": bandwidth,
                "n_boot": n_boot,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    bt = _load_module("backtest", "02_backtest.py")
    pt = _load_module("portfolio_tables", "04_portfolio_tables.py")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # (z, cost_bp, sample, block) cells: headline thresholds, then robustness.
    cells: list[tuple[float, float, str, int]] = [
        (z, BASELINE_COST_BP, "all_days", PRIMARY_BLOCK) for z in Z_THRESHOLDS
    ]
    cells += [(1.0, c, "all_days", PRIMARY_BLOCK) for c in COST_ROBUSTNESS_BP]
    cells += [
        (1.0, BASELINE_COST_BP, "all_days", b)
        for b in BLOCK_SENSITIVITY
        if b != PRIMARY_BLOCK
    ]
    cells += [(1.0, BASELINE_COST_BP, "post_warmup", PRIMARY_BLOCK)]

    rows: list[dict] = []
    cache: dict[tuple[float, float], tuple[pd.Series, pd.Series]] = {}
    for z, cost_bp, sample, block in cells:
        key = (z, cost_bp)
        if key not in cache:
            cache[key] = books(pt, bt, z, cost_bp)
        eg, simple = cache[key]
        if sample == "post_warmup":
            live = (eg != 0.0) | (simple != 0.0)
            if not bool(live.any()):
                continue
            start = live.idxmax()
            eg, simple = eg.loc[start:], simple.loc[start:]
        r1 = eg.to_numpy(dtype=float)
        r2 = simple.to_numpy(dtype=float)
        bandwidth = nw_bandwidth(len(r1))
        print(
            f"z={z:g} kappa={cost_bp:g}bp sample={sample} block={block} "
            f"n={len(r1)} nw_lag={bandwidth} corr={np.corrcoef(r1, r2)[0, 1]:.3f}"
        )
        for row in bootstrap_cell(
            r1,
            r2,
            block=block,
            n_boot=args.n_boot,
            bandwidth=bandwidth,
            seed=args.seed,
        ):
            rows.append(
                {
                    "z_threshold": z,
                    "cost_bp": cost_bp,
                    "sample": sample,
                    "corr_eg_simple": float(np.corrcoef(r1, r2)[0, 1]),
                    **row,
                }
            )
            print(
                f"  {row['stat']}: {row['point']:+.4f} "
                f"(HAC SE {row['hac_se']:.4f}, boot SD {row['boot_sd']:.4f}, "
                f"t {row['t_stat']:+.2f}) "
                f"p_stud {row['p_boot_studentized']:.4f} "
                f"p_pct {row['p_boot_percentile']:.4f} "
                f"p_hac {row['p_hac_normal']:.4f} "
                f"CI-pct [{row['ci95_percentile_lo']:+.4f}, "
                f"{row['ci95_percentile_hi']:+.4f}]"
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV.relative_to(ROOT)} ({len(out)} rows)")


if __name__ == "__main__":
    main()
