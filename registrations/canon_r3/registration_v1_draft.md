# Canon read 3 — the machine on the stock panel

**DRAFT REGISTRATION v1. UNSEALED. Not a seal, not a read.**
No ledger row exists, no controls receipt has been written against this draft,
and the gate refuses `--stage observed` while this file carries no sha256 anyone
has committed to. Statistics only; no vendor price appears in this file.

- Look: the third and last registered read of the Canon campaign (D-CANON-CHARTER
  part 3), a one-off exception to Stage R's closure under D-R6.
- Vintage: `20261004p`, not refreshed. Banked panel sha256 `7f13b1094b059eaa...`
- Feature file sha256 `23b7fbaf9968b2e3...`
- Pool manifest sha256 `559b912203db5b01...`
- Environment: python 3.13.3, numpy 2.5.3,
  pandas 3.0.5, pyarrow 25.0.1,
  scikit-learn 1.9.1
- Controls run: 2026-11-21T14:17:51Z, wall 3028 s.

## 1 The question

Does a machine, free to combine 25 price-and-volume characteristics of the
Sharadar common-stock cross-section, earn more as a traded book over
2001-12 to 2012-11 than the same pipeline earns in a world where the link
between a name and its own forward return has been destroyed?

## 2 The data and the window

Half A only, by the cross-sectional gate: every read of `bulk_stocks` passes
through `xs_triage.filter_to_half_a`, which filters at read time and then
asserts. 24,485,948 daily rows, maximum date loaded 2012-12-31, 3,774 global
sessions. The banked v2 half-A panel is read and never rewritten. Rows: every
in-universe row of the 168 formation months 1998-12 to 2012-11 — 168,000 rows,
1,000 names a month. `open`, `high`, `low` and `close` are never read.

## 3 The features (25, price and volume only)

REV1, MOM3_1, MOM6_1, MOM12_1, MOM12_7, SEAS12, REV5, VOL21, VOL63, VOL252, DOWNVOL63, MAX1, MIN1, SKEW63, UPFRAC63, HI52, LO52, AGEHI, HI63, DVTREND, AMIHUD21, VOLVOL63, PRC, BETA, LOGDV

All daily windows are trailing GLOBAL session windows ending at g, the session
before the formation month's month-end s_t; BETA and LOGDV are the banked
panel's own columns and therefore run through s_t. A window with fewer present
values than its minimum gives NaN. Each feature is then ranked within its month
and scaled to (-0.5, 0.5] by average rank / n - 0.5, so no raw value reaches the
model. Sector is not a feature.

Truncation, measured: at 1999-06, 2004-03, 2008-09 and 2012-11, every one of the
23 features this builder computes is identical when rebuilt from daily data cut
off after g — 0 NaN-pattern mismatches, maximum absolute delta 0.0.

| feature | share of rows non-NaN (before scaling) |
|---|---|
| AGEHI | 0.9829 |
| AMIHUD21 | 0.9997 |
| BETA | 0.9829 |
| DOWNVOL63 | 0.9965 |
| DVTREND | 0.9829 |
| HI52 | 0.9829 |
| HI63 | 0.9966 |
| LO52 | 0.9829 |
| LOGDV | 1.0000 |
| MAX1 | 0.9937 |
| MIN1 | 0.9937 |
| MOM12_1 | 0.9778 |
| MOM12_7 | 0.9778 |
| MOM3_1 | 0.9951 |
| MOM6_1 | 0.9900 |
| PRC | 1.0000 |
| REV1 | 0.9990 |
| REV5 | 1.0000 |
| SEAS12 | 0.9778 |
| SKEW63 | 0.9965 |
| UPFRAC63 | 0.9965 |
| VOL21 | 0.9997 |
| VOL252 | 0.9827 |
| VOL63 | 0.9965 |
| VOLVOL63 | 0.9966 |

## 4 The machine

Label, in a given world: that world's `fwd_ret`, residualised within the month on
[1, sector dummies, beta, log dollar volume] by
`xs_triage.residual_next_month_return`, then a within-month normal score. A
training target only; it never enters the feature matrix, and a spy model proves
the matrix is 25 columns wide.

Model: `HistGradientBoostingRegressor`, {"loss": "squared_error", "max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 10.0, "max_features": 0.5, "max_bins": 255, "early_stopping": false}. Three fits with
random_state [20261121, 20261122, 20261123]; the prediction is their mean. 8 threads. No tuning:
this is read 2's specification.

Walk: eleven fits at C = 2001-12 ... 2011-12. The fit at C trains on
formation months 1998-12 to C-2 inclusive and predicts C to C+11, so formation
month C-1 is the embargo. Scored formation months 2001-12 to
2012-11, 132 of them.

| cut C | train | score | train rows | score rows | fit s (3 seeds) |
|---|---|---|---|---|---|
| 2001-12 | 1998-12..2001-10 | 2001-12..2002-11 | 33417 | 12000 | 1.51 |
| 2002-12 | 1998-12..2002-10 | 2002-12..2003-11 | 45247 | 12000 | 1.65 |
| 2003-12 | 1998-12..2003-10 | 2003-12..2004-11 | 57142 | 12000 | 1.75 |
| 2004-12 | 1998-12..2004-10 | 2004-12..2005-11 | 69030 | 12000 | 1.88 |
| 2005-12 | 1998-12..2005-10 | 2005-12..2006-11 | 80913 | 12000 | 1.98 |
| 2006-12 | 1998-12..2006-10 | 2006-12..2007-11 | 92759 | 12000 | 2.13 |
| 2007-12 | 1998-12..2007-10 | 2007-12..2008-11 | 104603 | 12000 | 2.29 |
| 2008-12 | 1998-12..2008-10 | 2008-12..2009-11 | 116487 | 12000 | 2.38 |
| 2009-12 | 1998-12..2009-10 | 2009-12..2010-11 | 128454 | 12000 | 2.53 |
| 2010-12 | 1998-12..2010-10 | 2010-12..2011-11 | 140363 | 12000 | 2.68 |
| 2011-12 | 1998-12..2011-10 | 2011-12..2012-11 | 152258 | 12000 | 2.83 |

## 5 The book, the statistics, the verdict

Book: the prediction into harness v2's slope book (neutral to sector, beta and
log dollar volume, gross 2) — `build_series_v2` applies
`rank_to_normal_z(prediction, 'HIGH')` itself, which is the registered z. Costs
by `month_costs` at `COST_BASE` and `COST_STRESS`. Three monthly series over the
132 scored months: gross, net BASE, net STRESS. Months the harness skips are
counted and reported (0 skipped in every world of the pool).

Per series: mean, sd, t = mean / sd * sqrt(n), annualised Sharpe =
mean / sd * sqrt(12), turnover.

**The verdict, written before the read. One question, level 0.05.**

- **Key 1, information, on GROSS, both parts.**
  (a) `xs_triage`'s stationary bootstrap of months (mean block
  3.0, R 10000, seed 20261121) on the
  gross series less the median gross mean of the 60 worlds; one-sided p <= 0.05.
  (b) the gross t is equalled or exceeded by at most 2
  of the 60 worlds' gross t; p = (1 + count) / 61.
- **Key 2, materiality, on NET BASE.** Annualised Sharpe >= 0.5.
- NOTHING if key 1 fails; SIGNIFICANT_IMMATERIAL if key 1 holds and key 2 fails;
  SURVIVOR if both. NIL unless SURVIVOR.

## 6 The gate

`tools/run_canon_r3.py --stage observed` needs `--registration` and `--sha`,
verifies the file's sha256, checks the sha256 of every pinned artifact (the
banked panel, the feature file, the pool manifest, `xs_triage.py` and all nine
canon_r3 tools), demands a controls receipt whose `registration_sha` equals the
seal's, and takes an exclusive single-run lock. World 0 is reachable only from
that route: `build_world` demands a `VerifiedRegistration`, which only
`canon_r3_gate.load_registration` can mint, and re-verifies its file's sha256
before returning anything. The plant refuses world 0.

## 7 The measured controls (scrambled worlds only; no look was taken)

### 7.1 The pool
60 worlds, each a full eleven-fit walk and book. 132 months in
every world, 0 skipped. Gross t: median 0.2345,
sd 0.9474, min -1.9381, max 2.1568.
Net-BASE t: median -2.1279, sd 0.9572,
min -5.0150, max -0.2151.
Net-BASE annualised Sharpe across the worlds: min -1.512, max -0.065.

### 7.2 The known null
Key 1(a) applied to each world against the median of the other 59: share called
significant **0.0500** against the acceptance band
[0.01, 0.1] — `xs_triage.control_verdict` **PASS**.
Worlds the whole verdict calls SURVIVOR: **0** of
60. Verdict counts: {"NOTHING": 57, "SIGNIFICANT_IMMATERIAL": 3}.

### 7.3 The plants
Worlds 1 to 12, feature PRC replaced by the within-month rank scaling of
rho * z_y + sqrt(1 - rho^2) * e, with e from default_rng(9000 + 100k + j).
Key 1(b) for a planted world is taken against the 59 other unplanted worlds.

| rho | share 1(a) | share 1(b) | key 1 | share SURVIVOR | median gross Sharpe | median net-BASE Sharpe | median turnover |
|---|---|---|---|---|---|---|---|
| 0.005 | 0.000 | 0.083 | 0 of 12 | 0.000 | 0.078 | -0.696 | 2.595 |
| 0.010 | 0.000 | 0.000 | 0 of 12 | 0.000 | 0.151 | -0.622 | 2.613 |
| 0.020 | 0.667 | 0.667 | 7 of 12 | 0.000 | 0.724 | -0.075 | 2.648 |
| 0.040 | 1.000 | 1.000 | 12 of 12 | 1.000 | 2.409 | 1.589 | 2.740 |
| 0.080 | 1.000 | 1.000 | 12 of 12 | 1.000 | 6.364 | 5.574 | 2.825 |

### 7.4 MDE
the smallest rho at which key 1 passes in at least 10 of 12: **rho = 0.04**.
Median gross annualised Sharpe of its books: **2.409**.
Analytic gross annualised Sharpe for 80 per cent power on 132 months, one-sided
alpha 0.05: **0.7497**.

The measured MDE is about 3.2 times the
analytic figure, and that gap is the price of key 1 being two tests rather than
one: the analytic number is the 80-per-cent point for a single bootstrap on one
series, while key 1 also demands that the gross t beat 58 of 60 scrambled
worlds. The rho = 0.020 row, median gross Sharpe 0.724, sits
almost exactly on the analytic figure and passes key 1 in 7 of 12 — so the
analytic number is near this instrument's 50-per-cent point, not its 80.

### 7.5 Determinism
World 1 run twice: bit-identical on every series —
{"gross": true, "net_base": true, "net_stress": true, "turnover": true}.

### 7.6 Compute
Training rows per fit and fit seconds are in the table in section 4; the pool of
60 worlds plus the 60 planted runs, the known null and the
determinism pair took 3028 s in one process.

## 8 KNOWN BEFORE THE READ

- T-R1 and T-R1b read five of these characteristics singly on this window —
  MOM12_1, REV1, LOWVOL, HI52 and MAX, ten ledger rows across the two harness
  versions — and every one of the ten is **NOTHING**.
- Canon reads 1 and 2 were **NIL** (D-CANON-READ1, D-CANON-READ2).
- The charter states its expectation before any data: the machine on stocks
  "real but immaterial after costs".

## 9 LIMITS

- Costs are provisional constants in the research harness — 5 bps a side base,
  15 stress, 50/100 bps a year borrow, commission 0.0 — and **no certified
  equity cost row exists**. A SURVIVOR is therefore a research finding and no
  money test follows without one (D-P01-CLOSE's no-row-no-money-test gate).
- The delisting month's return is imputed 0.0: a name with a month-end close in
  m-1 and none in m is given exactly zero for m and flagged. `bulk_actions` is
  never read, so no liquidation or bankruptcy value enters.
- Sector and category are current-vintage attributes, not point-in-time. Sector
  is a control in harness v2 and category decides admission to the panel.
- No market capitalisation or shares-outstanding series exists in the snapshot;
  the universe is the top 1,000 by median 21-session dollar volume.
- The position is taken at the formation month-end close on features that stop a
  session earlier, except BETA and LOGDV, which run through s_t.
- The vendor's adjustment semantics are as the repo assumes: returns from
  `closeadj`, the price filter and dollar volume from `closeunadj`.
- Survivorship is evidenced by a 10-name sample of the 8,423 names delisted
  before 2015 (all PRESENT_AT_DEATH), not exhaustively verified.
- 132 scored months. The MDE in section 7.4 is what that buys.
- Half B has no guard in this repository yet. One must be built before any
  confirmation, and the holdout (2021-01-01 onward) stays sealed.
