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
    author-block(
      [Tetiana Lemishko#super[\*]],
      [Balanced Research, France\
      #text(size: 8.5pt)[(affiliation at the time of writing)]],
    ),
    author-block(
      [Alexandre Landi#super[\*]],
      [SKEMA Business School, France],
    ),
    author-block(
      [Juliana Caicedo-Llano],
      [University of Évry Paris-Saclay, France\
      Regent's University London, UK],
    ),
  )

  #v(0.35em)
  #text(size: 8.5pt)[
    #super[\*] These authors contributed equally to this work (co-first authors).
  ]

  #v(0.4em)
  #text(size: 10pt)[August 2026]

  #v(0.3em)
  #text(size: 9pt)[
    Correspondence: Alexandre Landi —
    #link("mailto:alexandre1.landi@skema.edu")
  ]
]

#v(1em)

#align(center)[#text(weight: "bold")[Abstract]]
#v(0.3em)

Pairs trading exploits pricing differentials between related assets. A cointegration-based
pairs strategy selects pairs that share a statistically significant long-run relationship, so that
short-term divergences are more plausibly mean-reverting. This paper studies whether that
filter improves risk-adjusted performance in the foreign-exchange market. Using daily spot
prices for the seven most liquid USD crosses over 2007–2025, we compare a simple always-trade
pairs rule with an Engle–Granger cointegration screen on rolling 257/21-day train/test windows,
across z-score entry thresholds $plus.minus 1$, $plus.minus 2$, and $plus.minus 3$. Charging a
stylized 2~bp round-trip cost on position changes, the unlevered cointegration portfolio
earns about 0.5% annualized at $plus.minus 1$ (and 0.2–0.3% at wider bands) because capital is
rarely deployed: invested fraction is about 8% of pair-days at $plus.minus 1$. It still
delivers higher Sharpe and Sortino ratios than the simple benchmark at every threshold.
A paired block-bootstrap test of the portfolio Sharpe and Sortino differences finds a
statistically significant edge at the pre-committed headline threshold $plus.minus 1$;
at $plus.minus 2$ and $plus.minus 3$ the ranking is consistent in point estimates but not
statistically separable. The ranking is unchanged relative to frictionless (zero-cost)
results and survives a grid up to 5~bp. A companion that sizes each live pair to 3%
train-window volatility still ranks cointegration first on Sharpe, with about 4%
annualized return at about 8% book volatility.

#v(0.4em)
#text(size: 10pt)[
  *Keywords:* Cointegration, Pairs Trading, Mean Reversion, Forex, Currency Markets \
  *JEL classification:* C58, F31, G11, G15
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
thin in FX. This paper fills that gap for major currencies. The seven most liquid USD
crosses are natural candidates for relative-value analysis: they share a common dollar
numeraire, are tightly linked by trade and capital flows, and are disciplined by parity
relations such as purchasing-power parity and interest-rate parity
@rogoff1996 @fama1984 @engelwest2005. Those links make long-run co-movement among majors
*plausible*; they do not, by themselves, deliver a trading rule. We therefore keep the
empirical design statistical: we compare (i) a *simple* pairs strategy that always trades
the synthetic spread on every out-of-sample block with (ii) a *cointegration-based*
strategy that trades only when Engle–Granger cointegration is detected on the preceding
training window. The universe comprises all unordered pairs among seven USD-denominated
majors ($C(7,2) = 21$ pairs). Performance is evaluated with annualized return and
volatility, Sharpe, Sortino, and Calmar ratios, and maximum drawdown
@sharpe1966 @sortino1991 @young1991 on *unlevered* portfolios, together with invested-fraction
and active-day occupancy. Cumulative-return figures scale the cointegration path to equal
ex-post volatility for visual comparison only.

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
equilibrium. In FX that equilibrium has a natural economic reading via parity and common
macro drivers; we spell out that motivation—and what we do *not* estimate—in the next
section.

= Methodology

== Data

We use daily adjusted closes for EUR/USD, GBP/USD, USD/JPY, USD/CHF, USD/CAD,
AUD/USD, and NZD/USD from Yahoo Finance @yahoo2024, January 1, 2007 to December 31,
2025. Quotes with USD as numerator are inverted so every series is XXXUSD
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

== Economic motivation

Why should an Engle–Granger screen be more than a statistical filter among 21 correlated
USD crosses? Purchasing-power parity (PPP) links national price levels to exchange rates;
although real exchange rates mean-revert only slowly, the associated “PPP puzzle” still
implies that related nominal rates are not free to wander independently forever
@rogoff1996. Covered and uncovered interest-rate parity connect spot rates, forwards, and
interest differentials; even when uncovered interest parity fails as a short-horizon
forecast @fama1984, exchange rates remain asset prices tightly tied to macro
fundamentals @engelwest2005. Together with a shared USD numeraire and deep G10
liquidity, these parity and macro links make *long-run co-movement*—and therefore
cointegration-based relative-value trading—economically plausible in this universe
@dunis2011.

Importantly, we do *not* estimate PPP baskets, trade real-exchange-rate gaps, or form
positions from UIP residuals. The trading object remains a bivariate log-price spread among
unordered XXXUSD majors, gated by the Engle–Granger screen described below. Parity
relations motivate *where* we look; the pre-committed EG rule determines *when* we trade.
At this stage, parity-based screens are left as a natural extension.

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
(a stylized all-in estimate for liquid G10 spot, not a full bid–ask plus rollover model). Zero cost recovers the
frictionless horse race; 5~bp is a conservative upper bound. Swap/rollover is left as
a caveat (short test windows and frequent flats keep overnight exposure limited).

== Strategy implementation and portfolio construction

*Simple pairs.* Trade every testing window for every unordered pair, using training moments
only for the z-score (alphabetical OLS hedge, no cointegration gate).

*Cointegration-based pairs.* Trade a testing window only if the Engle–Granger screen
above accepts cointegration on the preceding training window; otherwise the pair is flat.

Pair returns are summed and divided by 21 to form a standardized portfolio. Main tables
report these series *unlevered* and, unless noted, *after* $kappa = 2$~bp costs. Occupancy is
the mean *invested fraction*: the average across days of (number of pairs with a nonzero
lagged signal)~$\/ 21$. *Active days* are the share of calendar days on which at least one
pair is live. Flat days enter the Sharpe denominator as zeros, so unlevered volatility is
occupancy-diluted. As an implementable *companion*, on each 257-day
training block we estimate pair spread volatility
$hat(sigma) = "std"(r_1 - r_2) sqrt(252)$ and, on the following 21-day test block, size every
live pair by $L = 0.03 \/ hat(sigma)$---the same $L$ for cointegration and simple (mean
$L approx 0.33$). Pair PnL is $L$ times the net equal-notional return (costs $kappa = 2$~bp
scale with $L$); flat days stay at 0. The companion book is the *sum* of the 21 pair PnLs,
not $1 \/ 21$. The 3% per-position cap is an arbitrary choice. We do not force the two books
to the same occupancy, so book volatility follows how many pairs are live (@tbl:livevol3).
Cumulative-return figures separately
rescale the cointegration path so that *daily* volatilities match the simple path
(@tbl:eqvol); those figures are visual aids only.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: none,
    inset: (x: 8pt, y: 5pt),
    table.hline(),
    [z-score], [Equal-vol scale $L$], [Ann.\ vol EG (%)], [Ann.\ vol simple (%)],
    table.hline(stroke: 0.5pt),
    [$plus.minus 1$], [4.50], [0.89], [4.02],
    [$plus.minus 2$], [4.75], [0.60], [2.85],
    [$plus.minus 3$], [5.39], [0.38], [2.06],
    table.hline(),
  ),
  caption: [Equal ex-post daily-vol scales used in cumulative-return figures only ($kappa = 2$~bp; replication sample).],
) <tbl:eqvol>

The 21 unordered pairs are not independent experiments. Every leg is a USD cross, and many
pairs share a second currency, so pair returns comove under common dollar and risk shocks.
Pair-level Sharpe ratios are therefore descriptive of the
cross-section, not twenty-one separate hypothesis tests. We do not apply multiple-testing
adjustments or clustered inference to the pair panel. Our primary evidence is the
*portfolio* comparison of Engle–Granger versus always-trade rules (equal weight $1 \/ 21$),
which already aggregates those dependent legs. On those two daily portfolio series we test
whether the Sharpe and Sortino differences are distinguishable from zero with a paired
circular block bootstrap that resamples calendar dates once and applies the same blocks to
both books, preserving contemporaneous cross-book correlation
@ledoitwolf2008. Block length equals the 21-day test window; we studentize the Sharpe
difference with a Newey–West HAC standard error @neweywest1987 (Bartlett kernel; lag chosen by the
standard $4(T\/100)^(2\/9)$ rule) and report a percentile interval for the Sortino
difference (no closed-form HAC under our downside definition: annualized return over the
standard deviation of strictly negative daily returns). Calmar ratios remain descriptive:
maximum drawdown is path-dependent, and block reshuffling would scramble that path.
$plus.minus 1$ is the pre-committed headline design; $plus.minus 2$ and $plus.minus 3$
are reported alongside it (@tbl:delta-ratios).

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
    [Annualized return (%)], [0.49], [$-$0.17],
    [Annualized volatility (%)], [0.89], [4.02],
    [Sharpe ratio], [0.55], [$-$0.04],
    [Sortino ratio], [0.63], [$-$0.05],
    [Calmar ratio], [0.20], [$-$0.01],
    [Maximum drawdown (%)], [$-$2.48], [$-$15.89],
    [Invested fraction (%)], [8.24], [51.27],
    [Active days (%)], [68.33], [94.58],
    table.hline(),
  ),
  caption: [Unlevered performance at $z^star = plus.minus 1$ after $kappa = 2$~bp RT costs (windows 257/21, 21 unordered pairs, sample through 2025). Occupancy from lagged signals (independent of $kappa$). Frictionless Sharpes are 0.62 (EG) and 0.01 (simple).],
) <tbl:z1>

#figure(
  image(figpath("fig03_cum_returns_z1.png"), width: 88%),
  caption: [Cumulative returns at $z^star = plus.minus 1$ (equal ex-post vol; $kappa = 2$~bp; visuals only).],
) <fig:cum-z1>

Unlevered and after costs, cointegration earns 0.49% annualized with an invested fraction of
8.2% of pair-days and 68% active days: most of the $1 \/ 21$ book is cash on a typical day.
Volatility and drawdown are correspondingly far lower than the always-trade rule, whose 2~bp
haircut turns its already thin $z^star = 1$ edge slightly negative. The equal-vol equity
curve rises more steadily than the simple path, which suffers deeper early-sample drawdowns
(notably around 2008 and 2011–12).

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
    [Annualized return (%)], [0.25], [0.18],
    [Annualized volatility (%)], [0.60], [2.85],
    [Sharpe ratio], [0.41], [0.06],
    [Sortino ratio], [0.34], [0.06],
    [Calmar ratio], [0.13], [0.01],
    [Maximum drawdown (%)], [$-$1.93], [$-$12.51],
    [Invested fraction (%)], [2.97], [19.25],
    [Active days (%)], [31.55], [78.70],
    table.hline(),
  ),
  caption: [Unlevered performance at $z^star = plus.minus 2$ after $kappa = 2$~bp RT costs. Occupancy from lagged signals (independent of $kappa$).],
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
    [Annualized return (%)], [0.18], [0.21],
    [Annualized volatility (%)], [0.38], [2.06],
    [Sharpe ratio], [0.48], [0.10],
    [Sortino ratio], [0.25], [0.07],
    [Calmar ratio], [0.21], [0.02],
    [Maximum drawdown (%)], [$-$0.86], [$-$9.27],
    [Invested fraction (%)], [0.95], [5.72],
    [Active days (%)], [11.10], [38.15],
    table.hline(),
  ),
  caption: [Unlevered performance at $z^star = plus.minus 3$ after $kappa = 2$~bp RT costs. Occupancy from lagged signals (independent of $kappa$).],
) <tbl:z3>

#figure(
  image(figpath("fig05_cum_returns_z3.png"), width: 88%),
  caption: [Cumulative returns at $z^star = plus.minus 3$ (equal ex-post vol; $kappa = 2$~bp; visuals only).],
) <fig:cum-z3>

@tbl:sharpe-sum and @tbl:mdd-sum summarize the threshold comparative at $kappa = 2$.
@tbl:cost-sens shows that raising $kappa$ compresses both Sharpes, but the EG advantage
remains positive through 5~bp at every threshold; at $plus.minus 1$ the simple book is already
near zero without costs and negative once $kappa >= 1$.

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: none,
    inset: (x: 8pt, y: 5pt),
    table.hline(),
    [z-score], [Cointegration Sharpe], [Simple Sharpe],
    table.hline(stroke: 0.5pt),
    [$plus.minus 1$], [0.55], [$-$0.04],
    [$plus.minus 2$], [0.41], [0.06],
    [$plus.minus 3$], [0.48], [0.10],
    table.hline(),
  ),
  caption: [Unlevered Sharpe ratios across thresholds after $kappa = 2$~bp RT costs.],
) <tbl:sharpe-sum>

@tbl:delta-ratios reports dependence-robust inference on those portfolio gaps after
$kappa = 2$~bp. At the headline threshold $plus.minus 1$, $Delta$Sharpe is $+0.59$
($p = 0.014$; 95% studentized CI $[+0.12, +1.07]$) and $Delta$Sortino is $+0.68$
($p = 0.018$; 95% percentile CI $[+0.12, +1.28]$). At $plus.minus 2$ and $plus.minus 3$
the cointegration book still leads in point estimates, but the gaps are not statistically
separable at conventional levels. The $plus.minus 1$ conclusion is unchanged across the
cost grid $kappa in {0,1,2,5}$ and across block lengths of 5, 21, and 63 days; a HAC
delta-method Sharpe test @jobsonkorkie1981 @memmel2003 agrees with the bootstrap
(replication package).

#figure(
  table(
    columns: 6,
    align: (left, left, right, right, right, left),
    stroke: none,
    inset: (x: 5pt, y: 4pt),
    table.hline(),
    [z], [Statistic], [$Delta$], [$p$], [95% CI], [Method],
    table.hline(stroke: 0.5pt),
    [$plus.minus 1$], [Sharpe], [$+$0.59], [0.014], [[+0.12, +1.07]], [Studentized],
    [], [Sortino], [$+$0.68], [0.018], [[+0.12, +1.28]], [Percentile],
    [$plus.minus 2$], [Sharpe], [$+$0.35], [0.145], [[−0.14, +0.83]], [Studentized],
    [], [Sortino], [$+$0.27], [0.270], [[−0.22, +0.73]], [Percentile],
    [$plus.minus 3$], [Sharpe], [$+$0.37], [0.089], [[−0.07, +0.82]], [Studentized],
    [], [Sortino], [$+$0.18], [0.332], [[−0.21, +0.46]], [Percentile],
    table.hline(),
  ),
  caption: [Portfolio $Delta$Sharpe and $Delta$Sortino (EG $-$ simple) on the unlevered $1 \/ 21$ book after $kappa = 2$~bp. Paired circular block bootstrap ($B = 10{,}000$; block length 21 days; Newey–West lag 9 for studentized Sharpe). Sortino uses the paper's downside definition (std of strictly negative daily returns) with a percentile interval. $plus.minus 1$ is the pre-committed headline design.],
) <tbl:delta-ratios>

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
    [$plus.minus 1$], [0], [0.62], [0.01],
    [], [1], [0.58], [$-$0.02],
    [], [2], [0.55], [$-$0.04],
    [], [5], [0.45], [$-$0.12],
    [$plus.minus 2$], [0], [0.45], [0.11],
    [], [1], [0.43], [0.08],
    [], [2], [0.41], [0.06],
    [], [5], [0.35], [$-$0.00],
    [$plus.minus 3$], [0], [0.50], [0.13],
    [], [1], [0.49], [0.11],
    [], [2], [0.48], [0.10],
    [], [5], [0.43], [0.06],
    table.hline(),
  ),
  caption: [Unlevered Sharpe sensitivity to round-trip cost $kappa$ (bp of pair notional).],
) <tbl:cost-sens>

@tbl:occupancy reports capital deployment. At $plus.minus 1$, the cointegration book has an
invested fraction of 8.2% of pair-days versus 51% for always-trade; the Engle–Granger gate
itself is on only 18.5% of pair-days. Occupancy falls further at wider thresholds. These
figures explain the tiny unlevered volatilities.

#figure(
  table(
    columns: 6,
    align: (left, left, right, right, right, right),
    stroke: none,
    inset: (x: 5pt, y: 4pt),
    table.hline(),
    [z], [Strategy], [Invested (%)], [Active days (%)], [Live $|$ active], [Gate (%)],
    table.hline(stroke: 0.5pt),
    [$plus.minus 1$], [Cointegration], [8.24], [68.33], [2.53], [18.55],
    [], [Simple], [51.27], [94.58], [11.38], [94.72],
    [$plus.minus 2$], [Cointegration], [2.97], [31.55], [1.97], [18.55],
    [], [Simple], [19.25], [78.70], [5.14], [94.72],
    [$plus.minus 3$], [Cointegration], [0.95], [11.10], [1.80], [18.55],
    [], [Simple], [5.72], [38.15], [3.15], [94.72],
    table.hline(),
  ),
  caption: [Portfolio occupancy from lagged signals (independent of $kappa$). Invested fraction is the mean across days of live pairs~$\/ 21$; active days are days with at least one live pair; live $|$ active is the mean number of live pairs on those days; gate occupancy is the share of pair-days with a non-NaN z-score (Engle–Granger pass). Occupancy explains why unlevered volatility is small.],
) <tbl:occupancy>

@tbl:livevol3 applies that rule after $kappa = 2$~bp at $z^star in {1,2,3}$. Both strategies
use the same 3% train-window sizing, so book volatility follows how many pairs are on. At $plus.minus 1$ the
cointegration Sharpe edge survives (0.47 vs 0.09) with 3.87% return at 8.29% book vol. Sortino
is higher than Sharpe for the cointegration companion (0.79 vs 0.47)---and higher than the
unlevered EG Sortino of 0.63---because Sharpe penalizes right-tail volatility; the simple book
does not show that gap (0.11 vs 0.09). Its about 22% vol and $-$68% drawdown is the cost of
sizing many live pairs the same way. At $plus.minus 2$ and $plus.minus 3$ cointegration still
leads on Sharpe; simple has higher raw return because more pairs are live.

#figure(
  table(
    columns: 8,
    align: (left, left, right, right, right, right, right, right),
    stroke: none,
    inset: (x: 4pt, y: 4pt),
    table.hline(),
    [z], [Strategy], [Ann.\ ret.\ (%)], [Ann.\ vol.\ (%)], [Sharpe], [Sortino], [Calmar], [MDD (%)],
    table.hline(stroke: 0.5pt),
    [$plus.minus 1$], [Cointegration], [3.87], [8.29], [0.47], [0.79], [0.20], [$-$18.9],
    [], [Simple], [1.91], [21.97], [0.09], [0.11], [0.03], [$-$68.0],
    [$plus.minus 2$], [Cointegration], [1.65], [4.15], [0.40], [0.35], [0.17], [$-$9.75],
    [], [Simple], [2.21], [14.42], [0.15], [0.17], [0.06], [$-$37.2],
    [$plus.minus 3$], [Cointegration], [1.24], [3.12], [0.40], [0.22], [0.12], [$-$10.6],
    [], [Simple], [2.17], [9.81], [0.22], [0.18], [0.08], [$-$28.0],
    table.hline(),
  ),
  caption: [Companion: each live pair sized to 3% train-window vol ($L = 0.03 \/ hat(sigma)$; $kappa = 2$~bp). Same rule on both books; book = sum of pair PnLs (not $1 \/ 21$). The 3% cap is an arbitrary choice. Book vol follows how many pairs are live. Mean $L approx 0.33$.],
) <tbl:livevol3>

= Conclusions and future research

This study asks whether an Engle–Granger cointegration filter improves FX pairs trading
among seven liquid USD crosses. On rolling 257/21 windows and 21 unordered pairs, unlevered
cointegration returns are small (about 0.2–0.5% annualized after 2~bp) because occupancy is
low: at $plus.minus 1$ the invested fraction is about 8% of pair-days. The filter still
leads the always-trade benchmark on Sharpe and Sortino at $z^star in {1,2,3}$ in point
estimates. A paired block-bootstrap test finds that those risk-adjusted gaps are
statistically significant at the pre-committed headline threshold $plus.minus 1$, but not
separable at $plus.minus 2$ or $plus.minus 3$. The ranking matches the frictionless case
and remains positive through 5~bp. Sizing each live pair to 3% train-window volatility
raises cointegration to about 4% annualized return at about 8% book vol. Wider thresholds
further reduce unlevered volatility and drawdowns but shrink occupancy and the incremental
Sharpe benefit of the filter.

Natural extensions include a broader grid of window lengths and thresholds; parity-based
screens (PPP or UIP residuals) as alternatives or robustness checks to Engle–Granger;
overnight swap costs; and Johansen screens as further robustness.

#v(1em)
*Data and code availability.* Replication code, the frozen Yahoo-derived daily FX
sample (`data/fx_prices.parquet`), and this manuscript are available at
#link("https://github.com/QuantLandi/fx-cointegration"). Code, scripts, and data
are released under the MIT License; Typst source, PDF, figures, and bibliography
under `paper/` under CC BY 4.0. Pipeline instructions are in the repository README.

*Conflict of interest.* The authors declare no potential conflict of interest.

#bibliography("refs.bib", title: "References")
