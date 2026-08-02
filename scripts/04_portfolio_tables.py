"""Build JAE paper Tables 1–6 from EG and simple pair panels.

Requires:
  uv run python scripts/02_backtest.py --clear-panels
  uv run python scripts/02_backtest.py --simple --clear-panels

  uv run python scripts/04_portfolio_tables.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
COINT_DIR = OUTPUT_DIR / "coint"
SIMPLE_DIR = OUTPUT_DIR / "simple"
PANELS_EG = COINT_DIR / "panels"
PANELS_SIMPLE = SIMPLE_DIR / "panels"
METRICS_COINT = COINT_DIR / "metrics.csv"
METRICS_SIMPLE = SIMPLE_DIR / "metrics.csv"
PAPER_DIR = OUTPUT_DIR / "paper"
TABLES_DIR = PAPER_DIR / "tables"
PORTFOLIO_DIR = PAPER_DIR / "portfolio"
TARGETS_PATH = ROOT / "local" / "targets_jae.json"

Z_THRESHOLDS = (1.0, 2.0, 3.0)
N_PAIRS = 42


def _load_backtest():
    path = ROOT / "scripts" / "02_backtest.py"
    spec = importlib.util.spec_from_file_location("backtest", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def z_col(z: float) -> str:
    return f"z_{z:g}"


def list_pair_dirs(panels_root: Path) -> list[Path]:
    dirs = sorted(p for p in panels_root.iterdir() if p.is_dir())
    if len(dirs) != N_PAIRS:
        raise FileNotFoundError(
            f"Expected {N_PAIRS} pair dirs under {panels_root}, found {len(dirs)}. "
            "Run scripts/02_backtest.py (and --simple) first."
        )
    return dirs


def load_portfolio_returns(panels_root: Path, z: float) -> pd.Series:
    """Sum pair strategy returns / 42 (paper standardized portfolio)."""
    col = z_col(z)
    series: list[pd.Series] = []
    for pair_dir in list_pair_dirs(panels_root):
        path = pair_dir / "strategy_return.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if col not in df.columns:
            raise KeyError(f"Missing {col} in {path}")
        series.append(df[col].fillna(0.0))
    stacked = pd.concat(series, axis=1)
    stacked = stacked.fillna(0.0)
    return stacked.sum(axis=1) / N_PAIRS


def metrics_row(rets: pd.Series, metrics_from_returns) -> dict[str, float]:
    m = metrics_from_returns(rets)
    return {
        "ann_return_pct": m["ann_return"] * 100.0,
        "ann_volatility_pct": m["ann_volatility"] * 100.0,
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "calmar": m["calmar"],
        "max_drawdown_pct": m["max_drawdown"] * 100.0,
    }


def leverage_for_equal_mdd(
    eg: pd.Series,
    target_mdd: float,
    metrics_from_returns,
    *,
    lo: float = 0.01,
    hi: float = 50.0,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> float:
    """Find scalar L such that |MDD(eg * L)| matches |target_mdd| (fraction).

    Needed because max DD on ``exp(cumsum)-1`` does not scale linearly with L.
    """
    target = abs(target_mdd)
    if target == 0.0:
        return 1.0

    def mdd_abs(level: float) -> float:
        return abs(metrics_from_returns(eg * level)["max_drawdown"])

    # Expand upper bound if needed
    for _ in range(20):
        if mdd_abs(hi) >= target:
            break
        hi *= 2.0
    else:
        return hi

    best = hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        m = mdd_abs(mid)
        best = mid
        if abs(m - target) <= tol:
            return mid
        if m < target:
            lo = mid
        else:
            hi = mid
    return best


def refresh_pair_metrics(bt) -> None:
    """Rewrite metrics CSVs with Sortino/Calmar/MDD from existing panels."""
    for panels_root, out_path, strategy in (
        (PANELS_EG, METRICS_COINT, "eg"),
        (PANELS_SIMPLE, METRICS_SIMPLE, "simple"),
    ):
        rows: list[dict] = []
        train_window, test_window = bt.PAPER_WINDOW
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for pair_dir in list_pair_dirs(panels_root):
            legs = pair_dir.name.split("_")
            if len(legs) != 2:
                raise ValueError(f"Unexpected pair dir name: {pair_dir.name}")
            a1 = f"{legs[0].upper()}USD"
            a2 = f"{legs[1].upper()}USD"
            rets_df = pd.read_csv(
                pair_dir / "strategy_return.csv", index_col=0, parse_dates=True
            )
            sig_df = pd.read_csv(
                pair_dir / "signal.csv", index_col=0, parse_dates=True
            )
            for z in Z_THRESHOLDS:
                col = z_col(z)
                m = bt.metrics_from_returns(rets_df[col])
                rows.append(
                    {
                        "currency_1": a1,
                        "currency_2": a2,
                        "train_window": train_window,
                        "test_window": test_window,
                        "z_threshold": z,
                        **m,
                        "trades": int((sig_df[col] != 0).sum()),
                    }
                )
        out = pd.DataFrame(rows)
        out.to_csv(out_path, index=False)
        prices = pd.read_parquet(bt.PRICES_PATH)
        prices.index = pd.to_datetime(prices.index)
        bt.write_metrics_provenance(prices, out_path, strategy=strategy)
        print(f"Refreshed {out_path.relative_to(ROOT)} ({len(out)} rows)")


def main() -> None:
    bt = _load_backtest()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    refresh_pair_metrics(bt)

    table1_rows: list[dict] = []
    tables_24_rows: list[dict] = []
    paper_lev = {}
    if TARGETS_PATH.exists():
        targets = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        paper_lev = {
            float(k): float(v) for k, v in targets["table1_leverage_paper"].items()
        }

    for z in Z_THRESHOLDS:
        eg = load_portfolio_returns(PANELS_EG, z)
        simple = load_portfolio_returns(PANELS_SIMPLE, z)
        # Align calendars
        idx = eg.index.union(simple.index).sort_values()
        eg = eg.reindex(idx).fillna(0.0)
        simple = simple.reindex(idx).fillna(0.0)

        eg_unlev = metrics_row(eg, bt.metrics_from_returns)
        simple_m = metrics_row(simple, bt.metrics_from_returns)
        simple_mdd_frac = simple_m["max_drawdown_pct"] / 100.0
        leverage = leverage_for_equal_mdd(
            eg, simple_mdd_frac, bt.metrics_from_returns
        )
        eg_lev_rets = eg * leverage
        eg_m = metrics_row(eg_lev_rets, bt.metrics_from_returns)

        table1_rows.append(
            {
                "z_threshold": z,
                "leverage_computed": leverage,
                "leverage_paper": paper_lev.get(z, float("nan")),
                "mdd_eg_unlevered_pct": eg_unlev["max_drawdown_pct"],
                "mdd_simple_pct": simple_m["max_drawdown_pct"],
            }
        )
        for side, m in (("eg", eg_m), ("simple", simple_m)):
            tables_24_rows.append({"z_threshold": z, "strategy": side, **m})

        eg_lev_rets.to_csv(
            PORTFOLIO_DIR / f"returns_eg_z{z:g}.csv",
            header=["return"],
            index_label="date",
        )
        simple.to_csv(
            PORTFOLIO_DIR / f"returns_simple_z{z:g}.csv",
            header=["return"],
            index_label="date",
        )
        wealth = pd.DataFrame(
            {
                "eg_levered": np.exp(eg_lev_rets.cumsum()) - 1.0,
                "simple": np.exp(simple.cumsum()) - 1.0,
            }
        )
        wealth.to_csv(
            PORTFOLIO_DIR / f"cum_return_z{z:g}.csv", index_label="date"
        )

    table1 = pd.DataFrame(table1_rows)
    tables_24 = pd.DataFrame(tables_24_rows)
    table1.to_csv(TABLES_DIR / "table1_leverage.csv", index=False)
    tables_24.to_csv(TABLES_DIR / "tables_2_4_by_z.csv", index=False)

    # Table 5 / 6 convenience extracts
    wide = tables_24.pivot(index="z_threshold", columns="strategy")
    t5 = pd.DataFrame(
        {
            "z_threshold": wide.index,
            "sharpe_eg": wide[("sharpe", "eg")].values,
            "sharpe_simple": wide[("sharpe", "simple")].values,
        }
    )
    t6 = pd.DataFrame(
        {
            "z_threshold": wide.index,
            "max_drawdown_eg_pct": wide[("max_drawdown_pct", "eg")].values,
            "max_drawdown_simple_pct": wide[("max_drawdown_pct", "simple")].values,
        }
    )
    t5.to_csv(TABLES_DIR / "table5_sharpe.csv", index=False)
    t6.to_csv(TABLES_DIR / "table6_mdd.csv", index=False)

    print(f"Saved tables under {TABLES_DIR.relative_to(ROOT)}/")
    print(f"Saved portfolio series under {PORTFOLIO_DIR.relative_to(ROOT)}/")
    print("\nTable 1 (leverage):")
    print(table1.to_string(index=False))
    print("\nTables 2–4 (equal-MDD):")
    print(tables_24.to_string(index=False))


if __name__ == "__main__":
    main()
