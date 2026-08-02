"""Generate paper figures under outputs/paper/figures/.

  Fig 1 — train/test schematic (from SSRN PDF)
  Fig 2 — Sharpe heatmap (annotated, PuOr)
  Figs 3–5 — equal-vol cumulative returns (z = ±1, ±2, ±3)

Requires:
  uv run python scripts/04_portfolio_tables.py
  uv run python scripts/05_plot_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
PAPER_DIR = OUTPUT_DIR / "paper"
FIG_DIR = PAPER_DIR / "figures"
PORTFOLIO_DIR = PAPER_DIR / "portfolio"
METRICS_EG = OUTPUT_DIR / "coint" / "metrics.csv"

CURRENCIES = (
    "AUDUSD",
    "CADUSD",
    "CHFUSD",
    "EURUSD",
    "GBPUSD",
    "JPYUSD",
    "NZDUSD",
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
BASELINE_COST_BP = 2.0


def savefig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path.relative_to(ROOT)}")


def fig1_schematic() -> None:
    """Extract the schematic embedded in the SSRN/JAE PDF."""
    src_pdf = ROOT.parent / "pdfs" / (
        "lemishko_landi_caicedo-llano_2024_cointegration_forex_pairs_trading.pdf"
    )
    out = FIG_DIR / "fig01_train_test_schematic.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not src_pdf.exists():
        print(f"Skip fig01: missing source PDF {src_pdf}")
        return
    import fitz

    pdf = fitz.open(src_pdf)
    page = pdf[22]
    imgs = page.get_images(full=True)
    if not imgs:
        raise RuntimeError(f"No images on page 23 of {src_pdf.name}")
    xref = imgs[0][0]
    pix = fitz.Pixmap(pdf, xref)
    if pix.n - pix.alpha > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    pix.save(str(out))
    print(f"Wrote {out.relative_to(ROOT)} (extracted from PDF)")


def _heatmap(
    df: pd.DataFrame,
    title: str,
    filename: str,
    cmap: str = "PuOr_r",
    *,
    annotate: bool = False,
    decimals: int = 1,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    data = df.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isfinite(val):
                    continue
                rgba = im.cmap(im.norm(val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "white" if luminance < 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.{decimals}f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )
    fig.tight_layout()
    savefig(fig, filename)


def fig2_sharpe_heatmap() -> None:
    if not METRICS_EG.exists():
        raise FileNotFoundError(f"Missing {METRICS_EG}; run 04_portfolio_tables.py")
    m = pd.read_csv(METRICS_EG)
    if "cost_bp" in m.columns:
        m = m[m["cost_bp"] == BASELINE_COST_BP]
    sub = m[m["z_threshold"] == 1.0].copy()
    sub["r"] = sub["currency_1"].map(LEG_LABEL)
    sub["c"] = sub["currency_2"].map(LEG_LABEL)
    labels = [LEG_LABEL[x] for x in CURRENCIES]
    mat = pd.DataFrame(np.nan, index=labels, columns=labels)
    for _, row in sub.iterrows():
        mat.loc[row["r"], row["c"]] = float(row["sharpe"])
    _heatmap(
        mat,
        f"Figure 2: Sharpe (κ = {BASELINE_COST_BP:g} bp RT)",
        "fig02_sharpe_heatmap.png",
        annotate=True,
        decimals=1,
    )


def fig3_5_cum_returns() -> None:
    for z, fig_n in zip(Z_THRESHOLDS, (3, 4, 5), strict=True):
        path = PORTFOLIO_DIR / f"cum_return_z{z:g}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run 04_portfolio_tables.py")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        eg_col = "eg_equalvol" if "eg_equalvol" in df.columns else "eg_levered"
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(df.index, df[eg_col], label="Cointegration-based (equal vol)")
        ax.plot(df.index, df["simple"], label="Simple pairs")
        ax.set_title(
            f"Figure {fig_n}: Cumulative returns at z=±{z:g} "
            f"(equal ex-post vol, κ = {BASELINE_COST_BP:g} bp RT)"
        )
        ax.set_ylabel("Cumulative return")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelrotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        savefig(fig, f"fig{fig_n:02d}_cum_returns_z{z:g}.png")


def main() -> None:
    fig1_schematic()
    fig2_sharpe_heatmap()
    fig3_5_cum_returns()
    print(f"Done. Figures in {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
