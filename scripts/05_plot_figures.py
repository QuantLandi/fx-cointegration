"""Generate JAE paper Figures 1–18 under outputs/paper/figures/.

Requires panels + portfolio CSVs from 02 and 04:

  uv run python scripts/04_portfolio_tables.py
  uv run python scripts/05_plot_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
PAPER_DIR = OUTPUT_DIR / "paper"
FIG_DIR = PAPER_DIR / "figures"
PORTFOLIO_DIR = PAPER_DIR / "portfolio"
PRICES_PATH = ROOT / "data" / "fx_prices.parquet"
METRICS_EG = OUTPUT_DIR / "coint" / "metrics.csv"

# Heatmap axis order (alphabetical by ISO code).
CURRENCIES = (
    "AUDUSD",
    "CADUSD",
    "CHFUSD",
    "EURUSD",
    "GBPUSD",
    "JPYUSD",
    "NZDUSD",
)
# Paper Figs 2–8 order: AUD, EUR, GBP, NZD, JPY, CHF, CAD.
FIG_CURRENCY_ORDER = (
    "AUDUSD",
    "EURUSD",
    "GBPUSD",
    "NZDUSD",
    "JPYUSD",
    "CHFUSD",
    "CADUSD",
)
LEG_LABEL = {
    "AUDUSD": "AUD",
    "CADUSD": "CAD",
    "CHFUSD": "CHF",
    "EURUSD": "EUR",
    "GBPUSD": "GBP",
    "JPYUSD": "JPY",
    "NZDUSD": "NZD",
}
Z_THRESHOLDS = (1.0, 2.0, 3.0)


def savefig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path.relative_to(ROOT)}")


def fig1_schematic() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("Figure 1: Rolling train (257d) / test (21d) windows")
    y_train, y_test = 1.8, 0.7
    for k in range(6):
        t0 = 0.3 + k * 1.2
        ax.add_patch(
            plt.Rectangle((t0, y_train), 2.0, 0.6, fill=False, linewidth=1.2)
        )
        ax.text(t0 + 1.0, y_train + 0.3, "train", ha="center", va="center", fontsize=8)
        ax.add_patch(
            plt.Rectangle((t0 + 2.0, y_test), 0.35, 0.6, fill=True, alpha=0.25)
        )
        ax.text(
            t0 + 2.175,
            y_test + 0.3,
            "test",
            ha="center",
            va="center",
            fontsize=7,
            rotation=90,
        )
    ax.text(8.5, 2.7, "time →", ha="center")
    savefig(fig, "fig01_train_test_schematic.png")


def fig2_8_currency_zscores(prices: pd.DataFrame) -> None:
    for i, sym in enumerate(FIG_CURRENCY_ORDER, start=2):
        r = np.log(prices[sym]).diff().dropna()
        z = (r - r.mean()) / r.std()
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(z.index, z.values, linewidth=0.6)
        axes[0].set_title(f"{LEG_LABEL[sym]} return z-score")
        axes[0].set_ylim(-4.5, 4.5)
        axes[0].axhline(0, color="black", linewidth=0.5)
        axes[1].hist(z.values, bins=50, color="steelblue", edgecolor="none")
        axes[1].set_title(f"{LEG_LABEL[sym]} z-score distribution")
        fig.suptitle(f"Figure {i}: {LEG_LABEL[sym]}")
        fig.tight_layout()
        savefig(fig, f"fig{i:02d}_{LEG_LABEL[sym].lower()}_return_zscore.png")


def _residual_moments(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [LEG_LABEL[c] for c in CURRENCIES]
    skew = pd.DataFrame(np.nan, index=labels, columns=labels)
    kurt = pd.DataFrame(np.nan, index=labels, columns=labels)
    logs = np.log(prices[list(CURRENCIES)])
    for y in CURRENCIES:
        for x in CURRENCIES:
            if y == x:
                continue
            fit = sm.OLS(logs[y], sm.add_constant(logs[x])).fit()
            resid = fit.resid
            skew.loc[LEG_LABEL[y], LEG_LABEL[x]] = float(resid.skew())
            kurt.loc[LEG_LABEL[y], LEG_LABEL[x]] = float(resid.kurtosis() + 3.0)
    return skew, kurt


def _heatmap(df: pd.DataFrame, title: str, filename: str, cmap: str = "RdBu_r") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(df.to_numpy(dtype=float), cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    savefig(fig, filename)


def fig9_10_residual_heatmaps(prices: pd.DataFrame) -> None:
    skew, kurt = _residual_moments(prices)
    _heatmap(skew, "Figure 9: Residual skewness", "fig09_residual_skewness.png")
    _heatmap(kurt, "Figure 10: Residual kurtosis", "fig10_residual_kurtosis.png", "viridis")


def fig11_15_pair_metric_heatmaps() -> None:
    if not METRICS_EG.exists():
        raise FileNotFoundError(f"Missing {METRICS_EG}; run 04_portfolio_tables.py")
    m = pd.read_csv(METRICS_EG)
    # One heatmap per metric at z=1 (paper heatmaps are full-period pair stats;
    # we use EG-filtered strategy metrics at the paper baseline threshold).
    sub = m[m["z_threshold"] == 1.0].copy()
    sub["r"] = sub["currency_1"].map(LEG_LABEL)
    sub["c"] = sub["currency_2"].map(LEG_LABEL)
    labels = [LEG_LABEL[x] for x in CURRENCIES]
    specs = [
        (11, "ann_return", "Annualized return", True, "fig11_ann_return_heatmap.png"),
        (12, "ann_volatility", "Annualized volatility", True, "fig12_ann_vol_heatmap.png"),
        (13, "sharpe", "Sharpe", False, "fig13_sharpe_heatmap.png"),
        (14, "sortino", "Sortino", False, "fig14_sortino_heatmap.png"),
        (15, "calmar", "Calmar", False, "fig15_calmar_heatmap.png"),
    ]
    for fig_n, col, title, as_pct, fname in specs:
        mat = pd.DataFrame(np.nan, index=labels, columns=labels)
        for _, row in sub.iterrows():
            val = float(row[col])
            if as_pct:
                val *= 100.0
            mat.loc[row["r"], row["c"]] = val
        _heatmap(mat, f"Figure {fig_n}: {title}", fname)


def fig16_18_cum_returns() -> None:
    for z, fig_n in zip(Z_THRESHOLDS, (16, 17, 18), strict=True):
        path = PORTFOLIO_DIR / f"cum_return_z{z:g}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run 04_portfolio_tables.py")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(df.index, df["eg_levered"], label="Cointegration-based (equal MDD)")
        ax.plot(df.index, df["simple"], label="Simple pairs")
        ax.set_title(
            f"Figure {fig_n}: Cumulative returns at z=±{z:g} (equal risk)"
        )
        ax.set_ylabel("Cumulative return")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        savefig(fig, f"fig{fig_n:02d}_cum_returns_z{z:g}.png")


def main() -> None:
    prices = pd.read_parquet(PRICES_PATH)
    prices.index = pd.to_datetime(prices.index)
    fig1_schematic()
    fig2_8_currency_zscores(prices)
    fig9_10_residual_heatmaps(prices)
    fig11_15_pair_metric_heatmaps()
    fig16_18_cum_returns()
    print(f"Done. Figures in {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
