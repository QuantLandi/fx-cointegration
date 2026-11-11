# FX cointegration pairs trading

Replication code and short *Finance Research Letters* manuscript for Lemishko,
Landi & Caicedo-Llano, *Does a Cointegration Filter Help in Forex Pairs Trading?*
(November 2026; frozen sample through 2025-12-31). Preprint landing page:
[SSRN](https://ssrn.com/abstract=4771108). If the SSRN PDF lags this repository,
treat `paper/main.pdf` (or the EM-identical `paper/letter.pdf`) and the frozen
parquet here as authoritative for current numbers.

**Manuscript layout under `paper/`:**
- **`main.typ` / `main.pdf`** — primary public short letter (FRL)
- **`letter.tex` / `letter.pdf` + `Figure_1.png`** — same letter in LaTeX (upload twin)
- **`working_paper.typ` / `working_paper.pdf`** — archived longer draft (prior IRFA-era rewrite)

Please cite this repository via GitHub **Cite this repository** /
[`CITATION.cff`](CITATION.cff).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```bash
git clone https://github.com/QuantLandi/fx-cointegration.git
cd fx-cointegration
uv sync
```

Frozen prices are in `data/fx_prices.parquet`. Re-download only if you need a new freeze:

```bash
uv run python scripts/01_download_prices.py
```

## Pipeline

From the repo root:

```bash
uv run python scripts/02_backtest.py --clear-panels           # EG → outputs/coint/
uv run python scripts/02_backtest.py --simple --clear-panels  # → outputs/simple/
uv run python scripts/03_compare_strategies.py               # → outputs/compare/
uv run python scripts/04_portfolio_tables.py                 # → outputs/paper/tables|portfolio
uv run python scripts/05_plot_figures.py                     # → outputs/paper/figures/
uv run python scripts/06_per_trade_1pct_vol.py --target-vol 0.03  # companion 3% vol per live pair
uv run python scripts/07_delta_sharpe_bootstrap.py            # portfolio ΔSharpe / ΔSortino inference
```

`04_portfolio_tables.py` also refreshes pair `metrics.csv` from existing panels
(including the `{0,1,2,5}` bp cost grid), so you need not re-run EG just for
metrics/portfolio tables. `07_delta_sharpe_bootstrap.py` reads those same panels
and writes `outputs/paper/tables/tables_delta_sharpe_bootstrap.csv` (no `02` re-run).

**Runtime:** full `02`–`07` on a laptop is typically on the order of **tens of
minutes** (Engle–Granger filters dominate; `--simple` is faster). Steps `03`–`07`
are quick once panels exist.

## Data

Daily Yahoo Finance FX spots, 2007-01-01 to 2025-12-31, seven USD crosses
(AUD, CAD, CHF, EUR, GBP, JPY, NZD), all quoted as XXXUSD.

## Method (sketch)

1. For each **undirected** pair (21 = C(7,2), alphabetical legs) and rolling
   train/test window, run an Engle–Granger–style **tradability filter**: both
   OLS orientations are tested; among passes, keep the clearer residual ADF
   (more negative t-stat) and map the hedge into `log_1 − β·log_2`.
   Level ADF tests stay ordinary (both legs must look I(1)). The residual
   stage uses an ordinary ADF *p*-cutoff of 0.05 as a **pre-committed tuning
   parameter** (Vidyamurthy-style rule), *not* a size-correct
   MacKinnon cointegration test. Formal EG critical values are stricter
   (≈ −3.36 vs ≈ −2.87 at 5% for N=2 with a constant); we keep the looser
   cutoff deliberately and do not claim “cointegration detected at 5%.”
   Under this rule the filter accepts 917/4683 training windows (19.6%).
2. If the filter passes, form that spread and standardize with train mean/SD for an
   OOS z-score.
3. Trade when |z| exceeds a threshold (long spread if z < −z*, short if z > z*).
   Signals are lagged two days.
4. Strategy return = signal × (r₁ − r₂) (**equal notional**, not β-hedged).
   Annualized Sharpe uses **all calendar days** (flat days as 0), so volatility
   is diluted when often out of market. Metrics `trades` = days with nonzero
   signal, not round-trips.
5. **Transaction costs:** round-trip κ bp of pair notional charged as
   `(κ/1e4)·|Δsignal|/2` (open/close = κ/2 each; flip = κ). Grid
   `{0, 1, 2, 5}` bp; headline **κ = 2**. Panels store gross returns; costs are
   applied when building metrics and portfolio tables.
6. **Paper portfolio:** sum the 21 pair daily returns, divide by 21. Main tables
   are **unlevered** and report **invested fraction** (mean live pairs / 21) and
   **active days** (share of dates with at least one live pair) from `signal.csv`.
   A companion table sizes **each live pair** to **3% annualized vol** from
   train-window std of the spread return (same *L* for EG and simple; book =
   **sum** of pair PnLs, not 1/21). Cumulative-return figures scale the EG path
   to **equal ex-post daily vol** vs simple (visuals only). Sortino uses the std
   of strictly negative daily returns; Calmar = ann return / |max DD|.

**Paper subset:** train=257, test=21, z* ∈ {1, 2, 3}, 21 undirected pairs,
cost κ ∈ {0, 1, 2, 5} bp (baseline 2). Simple benchmark: `--simple` (no EG gate).

## Outputs

```text
outputs/
  coint/
    metrics.csv (+ .meta.json)   # rows × z × cost_bp; panels are gross
    panels/{leg1}_{leg2}/        # prices, returns, spread, zscore, signal, …
  simple/
    metrics.csv (+ .meta.json)
    panels/{leg1}_{leg2}/
  compare/
    metrics.csv                  # EG vs simple join (ΔSharpe), keyed by cost_bp
  paper/
    tables/                      # unlevered / occupancy / 3% live-pair vol / cost-sensitivity CSVs
    portfolio/                   # daily / cumulative series at baseline κ=2
    figures/                     # fig01–fig05
```

Under EG, `spread` / `zscore` are NaN outside Engle–Granger-pass OOS blocks.
Under `--simple`, almost all OOS blocks are filled (NaN only before the first
window or if train std is zero). Strategy return panels are **gross** (zero cost);
flat days are 0. Net returns apply `(κ/1e4)·|Δsignal|/2` in metrics and `04`.

`outputs/` and `local/` are gitignored. Optional local checks (JAE table
tolerances, notebook spot checks) live under `local/`.

Manuscript: short FRL letter in `paper/main.typ` / `paper/main.pdf` (LaTeX twin
`paper/letter.tex` / `paper/letter.pdf`). Longer archived draft:
`paper/working_paper.typ`. Pipeline figures regenerate under `outputs/paper/figures`
and are copied into `paper/figures` as needed.

## License

- **Code, scripts, `pyproject.toml`, `uv.lock`, and `data/`:** [MIT](LICENSE)
- **Manuscript assets under `paper/`** (Typst source, PDF, figures, bibliography):
  [CC BY 4.0](paper/LICENSE)

The frozen parquet in `data/` is a derived replication sample built from Yahoo
Finance FX spots; underlying Yahoo quotes remain subject to Yahoo’s terms of use.
