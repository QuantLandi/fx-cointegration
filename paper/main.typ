// Cointegration-Based Strategies in Forex Pairs Trading
// Typst rewrite of Lemishko, Landi & Caicedo-Llano (2024)
// Figures: ../outputs/paper/figures/ (relative to this file)

#set document(
  title: "Cointegration-Based Strategies in Forex Pairs Trading",
  author: ("Tetiana Lemishko", "Alexandre Landi", "Juliana Caicedo-Llano"),
)
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm))
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#set cite(style: "american-psychological-association")
#show link: underline
#show figure.caption: set text(size: 9.5pt)
#show table.cell.where(y: 0): set text(weight: "bold")

#let figpath(name) = "figures/" + name

#let author-block(name, affil) = {
  [#name\
  #text(size: 9.5pt, affil)]
}

#align(center)[
  #text(size: 16pt, weight: "bold")[
    Cointegration-Based Strategies in Forex Pairs Trading
  ]

  #v(0.8em)

  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1em,
    author-block([Tetiana Lemishko], [Balanced Research, Antibes, France]),
    author-block([Alexandre Landi], [IBM, Paris, France]),
    author-block([Juliana Caicedo-Llano], [Regent's University London, UK]),
  )

  #v(0.4em)
  #text(size: 10pt)[November 14, 2024]

  #v(0.3em)
  #text(size: 9pt)[
    Correspondence: Tetiana Lemishko —
    #link("mailto:tetiana.lemishko@balanced-research.com")
  ]
]

#v(1em)

#align(center)[#text(weight: "bold")[Abstract]]
#v(0.3em)

Pairs trading exploits pricing differentials between related assets. A cointegration-based
pairs strategy selects pairs that share a statistically significant long-run relationship, so that
short-term divergences are more plausibly mean-reverting. This paper studies whether that
filter improves risk-adjusted performance in the foreign-exchange market. Using daily spot
prices for the seven most liquid USD crosses over 2007–2024, we compare a simple always-trade
pairs rule with an Engle–Granger cointegration screen on rolling 257/21-day train/test windows,
across z-score entry thresholds $plus.minus 1$, $plus.minus 2$, and $plus.minus 3$. Charging a
2~bp round-trip cost on position changes, the unlevered cointegration portfolio still
delivers higher Sharpe, Sortino, and Calmar ratios than the simple benchmark at every
threshold, with the largest Sharpe advantage at $plus.minus 1$. The ranking is unchanged
relative to frictionless (zero-cost) results and survives a grid up to 5~bp.

#v(0.4em)
#text(size: 10pt)[
  *Keywords:* Cointegration, Pairs Trading, Mean Reversion, Forex, Currency Markets
]

#v(0.8em)

= Introduction

Traders continually search for relative-value inefficiencies. Pairs trading is a leading example:
identify assets that usually move together, and trade temporary divergences in anticipation of
realignment @gatev2006 @krauss2017.

Conventional implementations lean on correlation or distance statistics and treat mean
reversion as a short-horizon regularity. Cointegration reframes the problem. If two
non-stationary price series share a stationary linear combination, deviations from that
equilibrium are statistically constrained and therefore more credible as trading signals
@engle1987 @vidyamurthy2004.

Despite a large equity literature, cointegration-based pairs trading remains comparatively
thin in FX. This paper fills that gap for major currencies. We compare (i) a *simple* pairs
strategy that always trades the synthetic spread on every out-of-sample block with (ii) a
*cointegration-based* strategy that trades only when Engle–Granger cointegration is detected
on the preceding training window. The universe comprises all unordered pairs among seven
USD-denominated majors ($C(7,2) = 21$ pairs). Performance is evaluated with annualized
return and volatility, Sharpe, Sortino, and Calmar ratios, and maximum drawdown
@sharpe1966 @sortino1991 @young1991 on *unlevered* portfolios. Cumulative-return figures
scale the cointegration path to equal ex-post volatility for visual comparison only.

Portfolio metrics and @fig:cum-z1–@fig:cum-z3 below are produced by the companion
replication code on a frozen Yahoo Finance sample.

= Literature review

@krauss2017 organizes pairs-trading research into distance, time-series, stochastic-control,
cointegration, and other approaches.

The distance method of @gatev2006 forms pairs by historical Euclidean proximity and
opens when spreads widen. It is simple and relatively robust to data-snooping, but the
static $L^2$ metric is outlier-sensitive and poorly adapted to regime shifts.

@elliott2005 model the spread as a mean-reverting Gaussian Markov chain in state space.
The framework is tractable, yet Gaussian assumptions can fail in FX, especially in stress
episodes.

@jurek2007 study stochastic control for arbitrageurs choosing between a mean-reverting
spread and a risk-free asset, obtaining closed-form policies under Ornstein–Uhlenbeck
uncertainty.

@vidyamurthy2004 provides the cointegration blueprint: pair preselection, Engle–Granger
tradability screening, and nonparametric entry/exit rules. @rad2015 implement distance
preselection plus Engle–Granger and find return profiles similar to pure distance methods.
@galenko2012 use multivariate cointegration weights on ETF baskets and document
mean-reverting portfolio returns, with the usual caveat that performance can weaken outside
the estimation regime.

Related applications include equity hedging via cointegration @burgess2003, gold–silver
parity @liu2003, inflation hedges @bampinas2015 @bampinas2016, and currency-portfolio
optimization @dunis2011. @huck2015 compare distance, stationarity, and cointegration on
S\&P 500 constituents and conclude that cointegration delivers more stable excess returns
after costs; our contribution is the analogous horse race in major FX. Ranking/forecasting
approaches @huck2009 @huck2010 are complementary but outside our design.

Relative to short-horizon correlation filters, cointegration anchors trades in a long-run
equilibrium with clearer economic content and fewer spurious linkages.

= Methodology

== Data

We use daily adjusted closes for EUR/USD, GBP/USD, USD/JPY, USD/CHF, USD/CAD,
AUD/USD, and NZD/USD from Yahoo Finance @yahoo2024, January 1, 2007 to January 1,
2024. Quotes with USD as numerator are inverted so every series is XXXUSD
(e.g.\ JPYUSD $= 1/"USDJPY"$) @poundsterlinglive2022. Analysis uses log prices.

Prices are processed in rolling blocks: a training window of $n = 257$ days (about one year)
followed by a testing window of $m = 21$ days (about one month). After each test block,
windows advance by $m$ days so the next train ends where the previous test ended
(@fig:schematic). The training sample is used to (i) screen for cointegration when required
and (ii) estimate the spread mean and standard deviation used in the out-of-sample z-score.
The simple strategy skips (i) but still uses the training moments.

#figure(
  image(figpath("fig01_train_test_schematic.png"), width: 85%),
  caption: [Iterative training and testing windows over the sample.],
) <fig:schematic>

== Cointegration analysis

We test long-run co-movement with the Engle–Granger procedure @engle1987. Johansen
methods can recover multiple relations in larger systems; for bivariate FX pairs we prefer
Engle–Granger for transparency and leave Johansen to future work.

*Step 1.* Augmented Dickey–Fuller (ADF) tests on each log-price series. Failure to reject
the unit-root null supports treating both series as $I(1)$.

*Step 2.* OLS long-run regression
$ y_t = alpha + beta x_t + epsilon_t $ <eq:ols>
with residual $ hat(epsilon)_t = y_t - hat(beta) x_t - hat(alpha) $.

*Step 3.* ADF test on $hat(epsilon)_t$. Rejecting the residual unit-root null is evidence of
cointegration: $y_t$ and $x_t$ share a stationary combination even though each is
non-stationary. The hedge ratio $hat(beta)$ defines the synthetic spread.

Cointegration is a *relation*, not a directed trading object, so we work with
$C(7,2) = 21$ unordered pairs. Legs are ordered alphabetically by ticker so that each
pair has a unique synthetic spread $log P^(1) - beta log P^(2)$. The Engle–Granger
residual can depend on which series is the regressand; as a fixed design choice we
therefore test *both* OLS orientations on each training window and, among those that
pass at 5%, retain the orientation with the clearer residual ADF statistic (more negative
$t$-stat), mapping that hedge into the alphabetical spread. If neither orientation passes,
the pair is flat for the following test window.

== Mean-reversion strategy

On each testing day the z-score of the spread is
$ Z_t = (hat(epsilon)_t - mu) / sigma $ <eq:z>
where $mu$ and $sigma$ are the training mean and standard deviation of
$hat(epsilon)_t = log P^(1)_t - beta log P^(2)_t$ (intercept cancels in the z-score).

Enter a long spread when $Z_t < -z^star$ and a short spread when $Z_t > z^star$, for
thresholds $z^star in {1,2,3}$. Signals are lagged two days to allow for execution delay:
$ R^"gross"_t = "Signal"_(t-2) times (r^(1)_t - r^(2)_t) $ <eq:pnl>
with equal notional on each leg (not $beta$-hedged PnL). Flat days enter the Sharpe
denominator as zeros.

*Transaction costs.* Let $s_t in {-1,0,1}$ denote the lagged signal. We charge a
round-trip cost of $kappa$ basis points of pair notional on position changes,
$ "cost"_t = (kappa \/ 10^4) times |s_t - s_(t-1)| \/ 2 $,
so that opening or closing (|Δ$s$|=1) costs $kappa\/2$~bp and a long–short flip
(|Δ$s$|=2) costs a full $kappa$~bp. Net returns are
$ R^"net"_t = R^"gross"_t - "cost"_t $.
We report the grid $kappa in {0,1,2,5}$, with *headline* results at $kappa = 2$~bp
(a conventional all-in estimate for liquid G10 spot). Zero cost recovers the
frictionless horse race; 5~bp is a conservative upper bound. Swap/rollover is left as
a caveat (short test windows and frequent flats keep overnight exposure limited).

== Strategy implementation and portfolio construction

*Simple pairs.* Trade every testing window for every unordered pair, using training moments
only for the z-score (alphabetical OLS hedge, no cointegration gate).

*Cointegration-based pairs.* Trade a testing window only if the Engle–Granger screen
above accepts cointegration on the preceding training window; otherwise the pair is flat.

Pair returns are summed and divided by 21 to form a standardized portfolio. Main tables
report these series *unlevered* and, unless noted, *after* $kappa = 2$~bp costs. Because the
cointegration book is often flat, its unlevered volatility is much smaller than the
always-trade benchmark, so raw return and drawdown levels are hard to compare by eye. As a
*companion* presentation only, we also rescale each strategy ex post by
$L = sigma^star \/ hat(sigma)$ to a common annualized volatility target
$sigma^star = 10%$ (@tbl:target10). We choose 10% as a round, conventional risk budget
(neither fitted to the sample nor tied to either strategy’s realized vol). Sharpe and
Sortino are invariant to this scale; return, maximum drawdown, and Calmar are not, because
compounded drawdowns are nonlinear in $L$. Cumulative-return figures separately rescale the
cointegration path so that *daily* volatilities match the simple path (@tbl:eqvol); those
figures are visual aids only.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: none,
    inset: (x: 8pt, y: 5pt),
    table.hline(),
    [z-score], [Equal-vol scale $L$], [Ann.\ vol EG (%)], [Ann.\ vol simple (%)],
    table.hline(stroke: 0.5pt),
    [$plus.minus 1$], [4.44], [0.92], [4.10],
    [$plus.minus 2$], [4.74], [0.62], [2.94],
    [$plus.minus 3$], [5.41], [0.40], [2.15],
    table.hline(),
  ),
  caption: [Equal ex-post daily-vol scales used in cumulative-return figures only ($kappa = 2$~bp; replication sample).],
) <tbl:eqvol>

= Results

== Descriptive analysis

Per-currency return paths and additional descriptive heatmaps are omitted for brevity.
@fig:sharpe-hm shows pair-level annualized Sharpe ratios at the baseline threshold
$z^star = 1$ after $kappa = 2$~bp costs. Dispersion across unordered pairs is large; a few
crosses (notably some GBP and JPY combinations in this sample) dominate the right tail of
risk-adjusted pair performance.

#figure(
  image(figpath("fig02_sharpe_heatmap.png"), width: 72%),
  caption: [Heatmap of annualized Sharpe ratios by unordered pair ($z^star = 1$, EG-filtered, $kappa = 2$~bp RT).],
) <fig:sharpe-hm>

== Cointegration-based strategy versus simple pairs

We compare *unlevered* portfolios at training/testing windows 257/21. Headline tables use
$kappa = 2$~bp; @tbl:cost-sens reports Sharpe across the full cost grid. @tbl:z1 and
@fig:cum-z1 give the $z^star = 1$ results (figures use equal ex-post vol scaling on net
returns).

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: none,
    inset: (x: 8pt, y: 5pt),
    table.hline(),
    [Metric], [Cointegration], [Simple],
    table.hline(stroke: 0.5pt),
    [Annualized return (%)], [0.56], [$-$0.04],
    [Annualized volatility (%)], [0.92], [4.10],
    [Sharpe ratio], [0.61], [$-$0.01],
    [Sortino ratio], [0.69], [$-$0.01],
    [Calmar ratio], [0.23], [$-$0.00],
    [Maximum drawdown (%)], [$-$2.48], [$-$15.89],
    table.hline(),
  ),
  caption: [Unlevered performance at $z^star = plus.minus 1$ after $kappa = 2$~bp RT costs (windows 257/21, 21 unordered pairs). Frictionless Sharpes are 0.67 (EG) and 0.04 (simple).],
) <tbl:z1>

#figure(
  image(figpath("fig03_cum_returns_z1.png"), width: 88%),
  caption: [Cumulative returns at $z^star = plus.minus 1$ (equal ex-post vol; $kappa = 2$~bp; visuals only).],
) <fig:cum-z1>

Unlevered and after costs, cointegration still earns a positive annualized return at far
lower volatility and drawdown than the always-trade rule, whose 2~bp haircut turns its
already thin $z^star = 1$ edge slightly negative. The equal-vol equity curve rises more
steadily than the simple path, which suffers deeper early-sample drawdowns (notably around
2008 and 2011–12).

Widening the entry band to $plus.minus 2$ (@tbl:z2, @fig:cum-z2) reduces trading intensity and
absolute returns for the cointegration book, but the filter retains a clear risk-adjusted edge
after costs.

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: none,
    inset: (x: 8pt, y: 5pt),
    table.hline(),
    [Metric], [Cointegration], [Simple],
    table.hline(stroke: 0.5pt),
    [Annualized return (%)], [0.26], [0.16],
    [Annualized volatility (%)], [0.62], [2.94],
    [Sharpe ratio], [0.42], [0.06],
    [Sortino ratio], [0.34], [0.06],
    [Calmar ratio], [0.14], [0.01],
    [Maximum drawdown (%)], [$-$1.93], [$-$12.51],
    table.hline(),
  ),
  caption: [Unlevered performance at $z^star = plus.minus 2$ after $kappa = 2$~bp RT costs.],
) <tbl:z2>

#figure(
  image(figpath("fig04_cum_returns_z2.png"), width: 88%),
  caption: [Cumulative returns at $z^star = plus.minus 2$ (equal ex-post vol; $kappa = 2$~bp; visuals only).],
) <fig:cum-z2>

At $plus.minus 3$ (@tbl:z3, @fig:cum-z3) both books trade less often; cointegration still leads on
Sharpe and related ratios after costs, with much smaller unlevered drawdowns.

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: none,
    inset: (x: 8pt, y: 5pt),
    table.hline(),
    [Metric], [Cointegration], [Simple],
    table.hline(stroke: 0.5pt),
    [Annualized return (%)], [0.19], [0.17],
    [Annualized volatility (%)], [0.40], [2.15],
    [Sharpe ratio], [0.47], [0.08],
    [Sortino ratio], [0.25], [0.06],
    [Calmar ratio], [0.22], [0.02],
    [Maximum drawdown (%)], [$-$0.86], [$-$9.27],
    table.hline(),
  ),
  caption: [Unlevered performance at $z^star = plus.minus 3$ after $kappa = 2$~bp RT costs.],
) <tbl:z3>

#figure(
  image(figpath("fig05_cum_returns_z3.png"), width: 88%),
  caption: [Cumulative returns at $z^star = plus.minus 3$ (equal ex-post vol; $kappa = 2$~bp; visuals only).],
) <fig:cum-z3>

@tbl:sharpe-sum and @tbl:mdd-sum summarize the threshold comparative at $kappa = 2$.
@tbl:cost-sens shows that raising $kappa$ compresses both Sharpes, but the EG advantage
remains positive through 5~bp at every threshold; at $plus.minus 1$ the simple book is already
near zero at 1~bp and negative thereafter.

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: none,
    inset: (x: 8pt, y: 5pt),
    table.hline(),
    [z-score], [Cointegration Sharpe], [Simple Sharpe],
    table.hline(stroke: 0.5pt),
    [$plus.minus 1$], [0.61], [$-$0.01],
    [$plus.minus 2$], [0.42], [0.06],
    [$plus.minus 3$], [0.47], [0.08],
    table.hline(),
  ),
  caption: [Unlevered Sharpe ratios across thresholds after $kappa = 2$~bp RT costs.],
) <tbl:sharpe-sum>

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: none,
    inset: (x: 8pt, y: 5pt),
    table.hline(),
    [z-score], [Cointegration MDD (%)], [Simple MDD (%)],
    table.hline(stroke: 0.5pt),
    [$plus.minus 1$], [$-$2.48], [$-$15.89],
    [$plus.minus 2$], [$-$1.93], [$-$12.51],
    [$plus.minus 3$], [$-$0.86], [$-$9.27],
    table.hline(),
  ),
  caption: [Unlevered maximum drawdowns by threshold after $kappa = 2$~bp RT costs.],
) <tbl:mdd-sum>

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: none,
    inset: (x: 8pt, y: 5pt),
    table.hline(),
    [z-score], [$kappa$ (bp)], [EG Sharpe], [Simple Sharpe],
    table.hline(stroke: 0.5pt),
    [$plus.minus 1$], [0], [0.67], [0.04],
    [], [1], [0.64], [0.02],
    [], [2], [0.61], [$-$0.01],
    [], [5], [0.52], [$-$0.08],
    [$plus.minus 2$], [0], [0.46], [0.10],
    [], [1], [0.44], [0.08],
    [], [2], [0.42], [0.06],
    [], [5], [0.36], [$-$0.01],
    [$plus.minus 3$], [0], [0.49], [0.10],
    [], [1], [0.48], [0.09],
    [], [2], [0.47], [0.08],
    [], [5], [0.42], [0.04],
    table.hline(),
  ),
  caption: [Unlevered Sharpe sensitivity to round-trip cost $kappa$ (bp of pair notional).],
) <tbl:cost-sens>

@tbl:target10 reports the same portfolios after ex-post scaling to 10% annualized
volatility ($kappa = 2$). At matched vol, cointegration’s higher Sharpe translates into higher
scaled annualized return (e.g.\ 6.11% vs $-$0.09% at $plus.minus 1$). Scaled maximum drawdowns are
large for both books — especially cointegration, which requires substantial leverage to
reach 10% vol from a low unlevered base — so Calmar need not preserve the unlevered ranking.
We therefore treat @tbl:target10 as a magnitude aid, not a replacement for the unlevered
metrics.

#figure(
  table(
    columns: 7,
    align: (left, left, right, right, right, right, right),
    stroke: none,
    inset: (x: 5pt, y: 4pt),
    table.hline(),
    [z], [Strategy], [Scale $L$], [Ann.\ ret.\ (%)], [Sharpe], [Calmar], [MDD (%)],
    table.hline(stroke: 0.5pt),
    [$plus.minus 1$], [Cointegration], [10.82], [6.11], [0.61], [0.15], [$-$41.1],
    [], [Simple], [2.44], [$-$0.09], [$-$0.01], [$-$0.00], [$-$34.4],
    [$plus.minus 2$], [Cointegration], [16.12], [4.23], [0.42], [0.10], [$-$41.9],
    [], [Simple], [3.40], [0.56], [0.06], [0.02], [$-$36.6],
    [$plus.minus 3$], [Cointegration], [25.11], [4.66], [0.47], [0.12], [$-$39.1],
    [], [Simple], [4.64], [0.77], [0.08], [0.02], [$-$36.5],
    table.hline(),
  ),
  caption: [Companion metrics at 10% target annualized volatility after $kappa = 2$~bp (ex-post $L = 0.10 \/ hat(sigma)$). Ann.\ volatility is 10% by construction; Sharpe matches the unlevered net table.],
) <tbl:target10>

= Conclusions and future research

This study asks whether an Engle–Granger cointegration filter improves FX pairs trading
among seven liquid USD crosses. On rolling 257/21 windows and 21 unordered pairs, the
*unlevered* cointegration portfolio outperforms the always-trade benchmark on Sharpe,
Sortino, and Calmar at $z^star in {1,2,3}$ after a 2~bp round-trip cost, with the strongest
Sharpe edge at $plus.minus 1$. The ranking matches the frictionless case and remains positive
through 5~bp. A companion 10% target-vol table makes return magnitudes easier to compare
while leaving Sharpe unchanged. Wider thresholds further reduce unlevered volatility and
drawdowns but shrink the incremental Sharpe benefit of the filter.

Natural extensions include a broader grid of window lengths and thresholds; stronger
economic motivation from parity relationships (PPP / interest-rate parity); accounting for
cross-pair dependence when interpreting the 21-pair panel; overnight swap costs; and
Johansen screens as robustness to the Engle–Granger design.

#v(1em)
*Conflict of interest.* The authors declare no potential conflict of interest.

#bibliography("refs.bib", title: "References")
