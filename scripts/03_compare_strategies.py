"""Join EG vs simple metrics on the paper grid.

Requires prior runs of:
  uv run python scripts/02_backtest.py --clear-panels
  uv run python scripts/02_backtest.py --simple --clear-panels

  uv run python scripts/03_compare_strategies.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
EG_PATH = OUTPUT_DIR / "coint" / "metrics.csv"
SIMPLE_PATH = OUTPUT_DIR / "simple" / "metrics.csv"
COMPARE_DIR = OUTPUT_DIR / "compare"
OUT_PATH = COMPARE_DIR / "metrics.csv"

KEYS = (
    "currency_1",
    "currency_2",
    "train_window",
    "test_window",
    "z_threshold",
)


def main() -> None:
    if not EG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EG_PATH}. Run: uv run python scripts/02_backtest.py"
        )
    if not SIMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SIMPLE_PATH}. Run: "
            "uv run python scripts/02_backtest.py --simple"
        )

    eg = pd.read_csv(EG_PATH)
    simple = pd.read_csv(SIMPLE_PATH)
    merged = eg.merge(
        simple,
        on=list(KEYS),
        how="inner",
        suffixes=("_eg", "_simple"),
    )
    if len(merged) != len(eg) or len(merged) != len(simple):
        raise ValueError(
            f"Join incomplete: eg={len(eg)} simple={len(simple)} "
            f"merged={len(merged)}"
        )

    merged["delta_sharpe"] = merged["sharpe_eg"] - merged["sharpe_simple"]
    merged["eg_wins"] = merged["delta_sharpe"] > 0

    out = merged[
        [
            *KEYS,
            "sharpe_eg",
            "sharpe_simple",
            "delta_sharpe",
            "eg_wins",
            "ann_return_eg",
            "ann_return_simple",
            "ann_volatility_eg",
            "ann_volatility_simple",
            "trades_eg",
            "trades_simple",
        ]
    ].sort_values("delta_sharpe", ascending=False)

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    by_z = out.groupby("z_threshold", sort=True).agg(
        n=("delta_sharpe", "size"),
        mean_delta_sharpe=("delta_sharpe", "mean"),
        eg_win_share=("eg_wins", "mean"),
    )
    print(f"Saved {OUT_PATH.relative_to(ROOT)} ({len(out)} configs)")
    print(
        f"Overall: mean delta_sharpe (EG-simple) = {out['delta_sharpe'].mean():.4f}; "
        f"EG wins = {out['eg_wins'].mean():.1%} of configs"
    )
    print(by_z.to_string())
    print("\nTop 5 by delta_sharpe:")
    print(
        out.head(5)[
            [
                "currency_1",
                "currency_2",
                "z_threshold",
                "sharpe_eg",
                "sharpe_simple",
                "delta_sharpe",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
