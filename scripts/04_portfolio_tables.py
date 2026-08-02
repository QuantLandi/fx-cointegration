"""Build unlevered portfolio tables for EG vs simple (21 pairs).

Applies the RT bp cost grid on panel signals (panels store gross returns).
Headline tables / equity curves use BASELINE_COST_BP (2 bp); also write the
full {0,1,2,5} sensitivity.

Requires:
  uv run python scripts/02_backtest.py --clear-panels
  uv run python scripts/02_backtest.py --simple --clear-panels

  uv run python scripts/04_portfolio_tables.py
"""

from __future__ import annotations

import importlib.util
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

Z_THRESHOLDS = (1.0, 2.0, 3.0)
N_PAIRS = 21
TARGET_ANN_VOL = 0.10


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


def load_portfolio_returns(
    panels_root: Path,
    z: float,
    *,
    cost_rt_bp: float,
    apply_transaction_costs,
) -> pd.Series:
    col = z_col(z)
    series: list[pd.Series] = []
    for pair_dir in list_pair_dirs(panels_root):
        rets_df = pd.read_csv(
            pair_dir / "strategy_return.csv", index_col=0, parse_dates=True
        )
        sig_df = pd.read_csv(pair_dir / "signal.csv", index_col=0, parse_dates=True)
        if col not in rets_df.columns or col not in sig_df.columns:
            raise KeyError(f"Missing {col} in {pair_dir}")
        net = apply_transaction_costs(rets_df[col], sig_df[col], cost_rt_bp)
        series.append(net.fillna(0.0))
    stacked = pd.concat(series, axis=1).fillna(0.0)
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


def equal_vol_scale(eg: pd.Series, simple: pd.Series) -> float:
    s_eg = float(eg.std(ddof=0))
    s_simple = float(simple.std(ddof=0))
    if s_eg == 0.0:
        return 1.0
    return s_simple / s_eg


def target_vol_scale(rets: pd.Series, target_ann_vol: float, metrics_from_returns) -> float:
    vol = metrics_from_returns(rets)["ann_volatility"]
    if vol == 0.0:
        return 1.0
    return target_ann_vol / vol


def refresh_pair_metrics(bt, panels_root: Path, metrics_path: Path, strategy: str) -> None:
    rows: list[dict] = []
    train_window, test_window = bt.PAPER_WINDOW
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    for pair_dir in list_pair_dirs(panels_root):
        legs = pair_dir.name.split("_")
        if len(legs) != 2:
            raise ValueError(f"Unexpected pair dir name: {pair_dir.name}")
        a1 = f"{legs[0].upper()}USD"
        a2 = f"{legs[1].upper()}USD"
        rets_df = pd.read_csv(
            pair_dir / "strategy_return.csv", index_col=0, parse_dates=True
        )
        sig_df = pd.read_csv(pair_dir / "signal.csv", index_col=0, parse_dates=True)
        for z in Z_THRESHOLDS:
            col = z_col(z)
            for cost_bp in bt.COST_BPS:
                net = bt.apply_transaction_costs(rets_df[col], sig_df[col], cost_bp)
                m = bt.metrics_from_returns(net)
                rows.append(
                    {
                        "currency_1": a1,
                        "currency_2": a2,
                        "train_window": train_window,
                        "test_window": test_window,
                        "z_threshold": z,
                        "cost_bp": cost_bp,
                        **m,
                        "trades": int((sig_df[col] != 0).sum()),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(metrics_path, index=False)
    prices = pd.read_parquet(bt.PRICES_PATH)
    prices.index = pd.to_datetime(prices.index)
    bt.write_metrics_provenance(prices, metrics_path, strategy=strategy)
    print(f"Refreshed {metrics_path.relative_to(ROOT)} ({len(out)} rows)")


def main() -> None:
    bt = _load_backtest()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    baseline = bt.BASELINE_COST_BP

    refresh_pair_metrics(bt, PANELS_EG, METRICS_COINT, "engle_granger")
    refresh_pair_metrics(bt, PANELS_SIMPLE, METRICS_SIMPLE, "simple_no_eg_screen")

    scale_rows: list[dict] = []
    tables_24_rows: list[dict] = []
    target_vol_rows: list[dict] = []
    cost_sens_rows: list[dict] = []

    for cost_bp in bt.COST_BPS:
        for z in Z_THRESHOLDS:
            eg = load_portfolio_returns(
                PANELS_EG,
                z,
                cost_rt_bp=cost_bp,
                apply_transaction_costs=bt.apply_transaction_costs,
            )
            simple = load_portfolio_returns(
                PANELS_SIMPLE,
                z,
                cost_rt_bp=cost_bp,
                apply_transaction_costs=bt.apply_transaction_costs,
            )
            idx = eg.index.union(simple.index).sort_values()
            eg = eg.reindex(idx).fillna(0.0)
            simple = simple.reindex(idx).fillna(0.0)

            eg_m = metrics_row(eg, bt.metrics_from_returns)
            simple_m = metrics_row(simple, bt.metrics_from_returns)

            for side, rets, m in (("eg", eg, eg_m), ("simple", simple, simple_m)):
                cost_sens_rows.append(
                    {
                        "cost_bp": cost_bp,
                        "z_threshold": z,
                        "strategy": side,
                        **m,
                    }
                )

            if cost_bp != baseline:
                continue

            vol_scale = equal_vol_scale(eg, simple)
            eg_equalvol = eg * vol_scale

            for side, rets, m_unlev in (
                ("eg", eg, eg_m),
                ("simple", simple, simple_m),
            ):
                lev = target_vol_scale(rets, TARGET_ANN_VOL, bt.metrics_from_returns)
                m_tgt = metrics_row(rets * lev, bt.metrics_from_returns)
                target_vol_rows.append(
                    {
                        "z_threshold": z,
                        "cost_bp": cost_bp,
                        "strategy": side,
                        "target_ann_vol_pct": TARGET_ANN_VOL * 100.0,
                        "scale": lev,
                        **m_tgt,
                        "sharpe_unlevered": m_unlev["sharpe"],
                    }
                )

            scale_rows.append(
                {
                    "z_threshold": z,
                    "cost_bp": cost_bp,
                    "equal_vol_scale": vol_scale,
                    "ann_vol_eg_pct": eg_m["ann_volatility_pct"],
                    "ann_vol_simple_pct": simple_m["ann_volatility_pct"],
                    "mdd_eg_pct": eg_m["max_drawdown_pct"],
                    "mdd_simple_pct": simple_m["max_drawdown_pct"],
                }
            )
            for side, m in (("eg", eg_m), ("simple", simple_m)):
                tables_24_rows.append(
                    {"z_threshold": z, "cost_bp": cost_bp, "strategy": side, **m}
                )

            eg.to_csv(
                PORTFOLIO_DIR / f"returns_eg_z{z:g}.csv",
                header=["return"],
                index_label="date",
            )
            simple.to_csv(
                PORTFOLIO_DIR / f"returns_simple_z{z:g}.csv",
                header=["return"],
                index_label="date",
            )
            wealth_eqvol = pd.DataFrame(
                {
                    "eg_equalvol": np.exp(eg_equalvol.cumsum()) - 1.0,
                    "simple": np.exp(simple.cumsum()) - 1.0,
                }
            )
            wealth_eqvol.to_csv(
                PORTFOLIO_DIR / f"cum_return_z{z:g}.csv", index_label="date"
            )
            wealth_eqvol.to_csv(
                PORTFOLIO_DIR / f"cum_return_equalvol_z{z:g}.csv",
                index_label="date",
            )

    scale = pd.DataFrame(scale_rows)
    tables_24 = pd.DataFrame(tables_24_rows)
    target_vol = pd.DataFrame(target_vol_rows)
    cost_sens = pd.DataFrame(cost_sens_rows)
    scale.to_csv(TABLES_DIR / "equal_vol_scale.csv", index=False)
    tables_24.to_csv(TABLES_DIR / "tables_2_4_by_z.csv", index=False)
    target_vol.to_csv(TABLES_DIR / "tables_target_vol_10pct.csv", index=False)
    cost_sens.to_csv(TABLES_DIR / "tables_cost_sensitivity.csv", index=False)

    # Gross (0) vs baseline side-by-side for the paper.
    gross = cost_sens[cost_sens["cost_bp"] == 0.0]
    net = cost_sens[cost_sens["cost_bp"] == baseline]
    before_after = gross.merge(
        net,
        on=["z_threshold", "strategy"],
        suffixes=("_gross", "_net"),
    )
    before_after.to_csv(TABLES_DIR / "tables_gross_vs_baseline_cost.csv", index=False)

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

    # Sharpe vs cost for the sensitivity figure/table.
    sharpe_cost = cost_sens.pivot_table(
        index=["z_threshold", "cost_bp"],
        columns="strategy",
        values="sharpe",
    ).reset_index()
    sharpe_cost.to_csv(TABLES_DIR / "table_sharpe_by_cost.csv", index=False)

    print(f"Saved tables under {TABLES_DIR.relative_to(ROOT)}/")
    print(f"Saved portfolio series under {PORTFOLIO_DIR.relative_to(ROOT)}/")
    print(f"\nHeadline cost_bp = {baseline:g} (equal-vol plot scales):")
    print(scale.to_string(index=False))
    print(f"\nTables (unlevered, cost_bp={baseline:g}):")
    print(tables_24.to_string(index=False))
    print("\nCompanion table (10% target ann. vol, ex-post):")
    print(target_vol.to_string(index=False))
    print("\nSharpe by cost (sensitivity):")
    print(sharpe_cost.to_string(index=False))


if __name__ == "__main__":
    main()
