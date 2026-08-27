"""Per-live-pair vol overlay, EG and simple. Train-window spread vol, kappa=2 bp.

  uv run python scripts/06_per_trade_1pct_vol.py
  uv run python scripts/06_per_trade_1pct_vol.py --target-vol 0.03
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT.parent / "artifacts"
TABLES_DIR = ROOT / "outputs" / "paper" / "tables"
PANELS_EG = ROOT / "outputs" / "coint" / "panels"
PANELS_SIMPLE = ROOT / "outputs" / "simple" / "panels"
PRICES_PATH = ROOT / "data" / "fx_prices.parquet"

TRAIN_WINDOW = 257
TEST_WINDOW = 21
Z_THRESHOLDS = (1.0, 2.0, 3.0)
COST_BP = 2.0
N_PAIRS = 21


def _load_bt():
    path = ROOT / "scripts" / "02_backtest.py"
    spec = importlib.util.spec_from_file_location("backtest", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def z_col(z: float) -> str:
    return f"z_{z:g}"


def list_pair_dirs(root: Path) -> list[Path]:
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if len(dirs) != N_PAIRS:
        raise FileNotFoundError(f"Expected {N_PAIRS} pair dirs under {root}")
    return dirs


def leverage_series(
    returns: pd.DataFrame, idx: pd.DatetimeIndex, target_vol: float
) -> pd.Series:
    """L = target_vol / train spread vol, constant on each test block."""
    lev = pd.Series(0.0, index=idx)
    n = len(idx)
    r1 = returns["return_1"].reindex(idx)
    r2 = returns["return_2"].reindex(idx)
    spread = (r1 - r2).to_numpy()
    for i in range(TRAIN_WINDOW, n - TEST_WINDOW + 1, TEST_WINDOW):
        train = spread[i - TRAIN_WINDOW : i]
        train = train[np.isfinite(train)]
        if len(train) < 2:
            continue
        sigma = float(np.std(train, ddof=1) * np.sqrt(252))
        if sigma <= 0.0:
            continue
        lev.iloc[i : i + TEST_WINDOW] = target_vol / sigma
    return lev


def load_side(
    panels_root: Path, idx: pd.DatetimeIndex, apply_tc, target_vol: float
):
    nets: dict[float, list[pd.Series]] = {z: [] for z in Z_THRESHOLDS}
    levs: list[pd.Series] = []
    live: dict[float, list[pd.Series]] = {z: [] for z in Z_THRESHOLDS}
    for d in list_pair_dirs(panels_root):
        rets = pd.read_csv(d / "returns.csv", index_col=0, parse_dates=True)
        sig = pd.read_csv(d / "signal.csv", index_col=0, parse_dates=True)
        gross_df = pd.read_csv(d / "strategy_return.csv", index_col=0, parse_dates=True)
        levs.append(leverage_series(rets, idx, target_vol))
        for z in Z_THRESHOLDS:
            col = z_col(z)
            g = gross_df[col].reindex(idx).fillna(0.0)
            s = sig[col].reindex(idx).fillna(0.0)
            nets[z].append(apply_tc(g, s, COST_BP))
            live[z].append((s != 0).astype(float))
    return nets, levs, live


def book(nets: list[pd.Series], levs: list[pd.Series]) -> pd.Series:
    parts = [lv * nt for lv, nt in zip(levs, nets)]
    return pd.concat(parts, axis=1).fillna(0.0).sum(axis=1)


def unlevered(nets: list[pd.Series]) -> pd.Series:
    return pd.concat(nets, axis=1).fillna(0.0).sum(axis=1) / N_PAIRS


def n_live(live_cols: list[pd.Series]) -> pd.Series:
    return pd.concat(live_cols, axis=1).fillna(0.0).sum(axis=1)


def fmt(x: float, nd: int = 2) -> str:
    if np.isnan(x):
        return "—"
    if x < 0:
        return f"−{abs(x):.{nd}f}"
    return f"{x:.{nd}f}"


def metrics_row(m: dict[str, float]) -> dict[str, float]:
    return {
        "ret": m["ann_return"] * 100.0,
        "vol": m["ann_volatility"] * 100.0,
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "calmar": m["calmar"],
        "mdd": m["max_drawdown"] * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-vol", type=float, default=0.01)
    args = parser.parse_args()
    target_vol = args.target_vol
    pct = int(round(target_vol * 100))
    stem = f"paper1_per_trade_{pct}pct_vol"

    bt = _load_bt()
    prices = pd.read_parquet(PRICES_PATH)
    prices.index = pd.to_datetime(prices.index)
    idx = prices.index.sort_values()

    print("Loading EG panels...", flush=True)
    eg_nets, eg_levs, eg_live = load_side(
        PANELS_EG, idx, bt.apply_transaction_costs, target_vol
    )
    print("Loading simple panels...", flush=True)
    sm_nets, sm_levs, sm_live = load_side(
        PANELS_SIMPLE, idx, bt.apply_transaction_costs, target_vol
    )

    mean_l_eg = float(pd.concat(eg_levs, axis=1).replace(0.0, np.nan).mean().mean())
    mean_l_sm = float(pd.concat(sm_levs, axis=1).replace(0.0, np.nan).mean().mean())

    rows = []
    books: dict[tuple[str, float], pd.Series] = {}
    for z in Z_THRESHOLDS:
        eg_b = book(eg_nets[z], eg_levs)
        sm_b = book(sm_nets[z], sm_levs)
        books[("eg", z)] = eg_b
        books[("simple", z)] = sm_b
        eg_n = n_live(eg_live[z])
        sm_n = n_live(sm_live[z])
        for side, b, nets, nlv in (
            ("eg", eg_b, eg_nets[z], eg_n),
            ("simple", sm_b, sm_nets[z], sm_n),
        ):
            ov = metrics_row(bt.metrics_from_returns(b))
            un = metrics_row(bt.metrics_from_returns(unlevered(nets)))
            rows.append(
                {
                    "z": z,
                    "side": side,
                    "ov": ov,
                    "un": un,
                    "mean_live": float(nlv.mean()),
                    "mean_live_on": float(nlv[nlv > 0].mean()) if (nlv > 0).any() else 0.0,
                }
            )

    ART.mkdir(parents=True, exist_ok=True)
    eg1 = books[("eg", 1.0)]
    sm1 = books[("simple", 1.0)]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(eg1.index, np.exp(eg1.cumsum()) - 1.0, color="#1f4e79", lw=1.4, label="EG")
    ax.plot(sm1.index, np.exp(sm1.cumsum()) - 1.0, color="#6b6b6b", lw=1.4, label="Simple")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Date")
    ax.set_title(
        rf"{pct}% vol per live pair, $z^\star=1$ ($\kappa=2$ bp)"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    cum_png = f"{stem}_cum.png"
    fig.savefig(ART / cum_png, dpi=160)
    plt.close(fig)

    def table(kind: str) -> str:
        lines = []
        for z in Z_THRESHOLDS:
            for side in ("eg", "simple"):
                r = next(x for x in rows if x["z"] == z and x["side"] == side)
                m = r[kind]
                lab = "EG" if side == "eg" else "Simple"
                lines.append(
                    f"| $\\pm {z:g}$ | {lab} | {fmt(m['ret'])} | {fmt(m['vol'])} | "
                    f"{fmt(m['sharpe'])} | {fmt(m['sortino'])} | {fmt(m['calmar'])} | "
                    f"{fmt(m['mdd'])} |"
                )
        return "\n".join(lines)

    def live_table() -> str:
        lines = []
        for z in Z_THRESHOLDS:
            for side in ("eg", "simple"):
                r = next(x for x in rows if x["z"] == z and x["side"] == side)
                lab = "EG" if side == "eg" else "Simple"
                lines.append(
                    f"| $\\pm {z:g}$ | {lab} | {fmt(r['mean_live'], 2)} | "
                    f"{fmt(r['mean_live_on'], 2)} |"
                )
        return "\n".join(lines)

    eg1r = next(x for x in rows if x["z"] == 1.0 and x["side"] == "eg")
    sm1r = next(x for x in rows if x["z"] == 1.0 and x["side"] == "simple")

    md = f"""# Paper 1 — {pct}% vol per live pair (EG and simple)

> Experiment, not a paper table. **Same rule on both books:** if a pair is on,
> scale it to {pct}% annualized vol using train-window $\\mathrm{{std}}(r_1-r_2)\\sqrt{{252}}$;
> if it is flat, PnL is 0. $L={target_vol}/\\hat\\sigma_{{\\mathrm{{train}}}}$ is fixed for the
> 21-day test block. Costs $\\kappa=2$ bp scale with $L$. Book = **sum** of pair PnL
> (not $1/21$).

![Cumulative {pct}% vol per live pair]({cum_png})

At $z^\\star=1$, EG earns {fmt(eg1r['ov']['ret'])}% annualized at {fmt(eg1r['ov']['vol'])}%
vol (Sharpe {fmt(eg1r['ov']['sharpe'])}, MDD {fmt(eg1r['ov']['mdd'])}%). Simple:
{fmt(sm1r['ov']['ret'])}% at {fmt(sm1r['ov']['vol'])}% vol (Sharpe {fmt(sm1r['ov']['sharpe'])},
MDD {fmt(sm1r['ov']['mdd'])}%). Simple has more live pairs, so higher book vol is expected.

Mean $L$ on blocks with a train vol (both sides share the same spread vol):
EG {fmt(mean_l_eg, 2)}, simple {fmt(mean_l_sm, 2)}.

## Overlay ({pct}% vol per live pair, $\\kappa=2$ bp)

| z | Strategy | Ann. ret. (%) | Ann. vol. (%) | Sharpe | Sortino | Calmar | MDD (%) |
|--:|----------|--------------:|--------------:|-------:|--------:|-------:|--------:|
{table("ov")}

## Unlevered $1/21$ book (paper baseline, same costs)

| z | Strategy | Ann. ret. (%) | Ann. vol. (%) | Sharpe | Sortino | Calmar | MDD (%) |
|--:|----------|--------------:|--------------:|-------:|--------:|-------:|--------:|
{table("un")}

## Average number of live pairs

| z | Strategy | Mean live (incl. zeros) | Mean live $|$ any live |
|--:|----------|------------------------:|----------------------:|
{live_table()}

**Source.** Existing panels. Script: `fx-cointegration/scripts/06_per_trade_1pct_vol.py`.
"""
    out = ART / f"{stem}.md"
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    mean_l = {"eg": mean_l_eg, "simple": mean_l_sm}
    for r in rows:
        csv_rows.append(
            {
                "z_threshold": r["z"],
                "strategy": r["side"],
                "target_pair_vol": target_vol,
                "cost_bp": COST_BP,
                "ann_return_pct": r["ov"]["ret"],
                "ann_volatility_pct": r["ov"]["vol"],
                "sharpe": r["ov"]["sharpe"],
                "sortino": r["ov"]["sortino"],
                "calmar": r["ov"]["calmar"],
                "max_drawdown_pct": r["ov"]["mdd"],
                "mean_live": r["mean_live"],
                "mean_live_given_active": r["mean_live_on"],
                "mean_L": mean_l[r["side"]],
            }
        )
    csv_path = TABLES_DIR / f"tables_per_trade_{pct}pct.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    for r in rows:
        print(
            f"z={r['z']:g} {r['side']:6s}  ret {r['ov']['ret']:.2f}  "
            f"vol {r['ov']['vol']:.2f}  sh {r['ov']['sharpe']:.2f}"
        )


if __name__ == "__main__":
    main()
