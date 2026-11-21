# Canon read 3 — the machine on the stock panel — SEALED REGISTRATION

**SEALED. The design below is fixed. The read is taken once, against this file's
own sha256, and nothing here changes afterwards.**
Statistics only; no vendor value appears in this file.

- Look: the third and last registered read of the Canon campaign
  (D-CANON-CHARTER part 3), a one-off exception to Stage R's closure under D-R6.
- Vintage `20261004p`, not refreshed. Half A only. Half B stays unopened and the
  holdout sealed, whatever the verdict.
- Environment: python 3.13.3, numpy 2.5.3,
  pandas 3.0.5, pyarrow 25.0.1,
  scikit-learn 1.9.1.
- Controls on this code: 2026-11-21T16:01:16Z, wall 3985 s.

## 1 The question

Does a machine, free to combine 25 price-and-volume characteristics of the
Sharadar common-stock cross-section, earn more as a traded book over
2001-12 to 2012-11 than the same pipeline earns in a world where
the link between a name and its own forward return has been destroyed?

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
before the formation month's month-end s_t. BETA and LOGDV are the banked
panel's own columns and so run through s_t. A window with fewer present values
than its minimum gives NaN. Each feature is then ranked within its month and
scaled to (-0.5, 0.5], so no raw value reaches the model. Sector is not a
feature.

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
training target only; it never enters the feature matrix.

Model: `HistGradientBoostingRegressor`, {"loss": "squared_error", "max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 10.0, "max_features": 0.5, "max_bins": 255, "early_stopping": false}. Three fits with
random_state [20261121, 20261122, 20261123]; the prediction is their mean. 8 threads. No tuning.

Walk: eleven fits at C = 2001-12 ... 2011-12. The fit at C trains on
formation months 1998-12 to C-2 inclusive and predicts C to C+11, so formation
month C-1 is the embargo. Scored formation months 2001-12 to
2012-11, 132 of them, each covered by exactly one cut.

| cut C | train | score | train rows | fit s (3 seeds) |
|---|---|---|---|---|
| 2001-12 | 1998-12..2001-10 | 2001-12..2002-11 | 33417 | 1.54 |
| 2002-12 | 1998-12..2002-10 | 2002-12..2003-11 | 45247 | 1.66 |
| 2003-12 | 1998-12..2003-10 | 2003-12..2004-11 | 57142 | 1.75 |
| 2004-12 | 1998-12..2004-10 | 2004-12..2005-11 | 69030 | 1.90 |
| 2005-12 | 1998-12..2005-10 | 2005-12..2006-11 | 80913 | 2.00 |
| 2006-12 | 1998-12..2006-10 | 2006-12..2007-11 | 92759 | 2.19 |
| 2007-12 | 1998-12..2007-10 | 2007-12..2008-11 | 104603 | 2.28 |
| 2008-12 | 1998-12..2008-10 | 2008-12..2009-11 | 116487 | 2.42 |
| 2009-12 | 1998-12..2009-10 | 2009-12..2010-11 | 128454 | 2.58 |
| 2010-12 | 1998-12..2010-10 | 2010-12..2011-11 | 140363 | 2.71 |
| 2011-12 | 1998-12..2011-10 | 2011-12..2012-11 | 152258 | 2.83 |

## 5 The worlds

World k >= 1 shifts each formation month's targets one position along that
month's own key order, keys drawn by `default_rng(7000 + k)` over the sorted
permaticker list. The month's cross-sectional distribution of `fwd_ret` is
preserved exactly, nothing moves in time, and a world is a pure function of
(panel, k). World 0 is the real world and is reachable only through the gate.

## 6 The book, the statistics, the verdict

Book: the prediction into harness v2's slope book (neutral to sector, beta and
log dollar volume, gross 2). Costs by `month_costs` at `COST_BASE` and
`COST_STRESS`. Three monthly series over the 132 scored months:
gross, net BASE, net STRESS.

Per series: mean, sd, t = mean / sd * sqrt(n), annualised Sharpe =
mean / sd * sqrt(12), turnover.

**ONE QUESTION, LEVEL 0.05.**

- **Key 1, information, on GROSS, both parts.**
  (a) the stationary bootstrap of months (mean block 3.0,
  R 10000, seed 20261121) on the gross series less the median
  gross mean of all 60 worlds; one-sided p <= 0.05.
  (b) the gross t is equalled or exceeded by at most
  2 of the 60 worlds' gross t; p = (1 + count) / 61.
- **Key 2, materiality, on NET BASE.** Annualised Sharpe >= 0.5.
- NOTHING if key 1 fails; SIGNIFICANT_IMMATERIAL if key 1 holds and key 2 fails;
  SURVIVOR if both. NIL unless SURVIVOR.

### The bars, as numbers, fixed before the read

| bar | value |
|---|---|
| key 1(a) null centre — the pool's median gross mean | +0.0001692370 (+1.6924 bps) |
| key 1(b) — the third-largest pool gross t, which the read's gross t must exceed | +1.770610 |
| key 2 — net BASE annualised Sharpe | 0.50 |

## 7 The rulings (each applied in the code this seal pins)

a. The value "at g" is the name's last observation at or before g, the v2
   builder's own idiom.
b. DVTREND is NaN when either median dollar volume is zero; zero-dollar-volume
   sessions stay inside its medians.
c. AGEHI counts global sessions from the maximum's session to g.
d. The rank scaling divides by the count of ranked (non-NaN) values.
e. The raw prediction is passed to `build_series_v2` with direction HIGH; the
   harness makes the normal score. These words match the code.
f. In the plant control key 1(b) is taken against the 59 other worlds with
   "at most 2". This is a control convention only.
g. For the read, key 1(a)'s centre is the median gross mean of all 60 worlds;
   the control used the other 59.
h. BETA and LOGDV are features and are also neutralised by the book.
i. The label is neutral to sector, beta and size, and no feature knows the
   sector: the question is whether price and volume predict that residual, which
   is narrower than predicting returns.
j. The book's cross-section averages 990.3 of 1,000 here against 982.8 in
   T-R1b, because the window starts at 1998-12.
k. Key 1 is voted on GROSS because a null world's turnover need not equal the
   real world's.
l. No smoothing, holding-period or turnover variant is added. One specification.

## 8 The measured controls (scrambled worlds only; no look was taken)

### The pool
60 worlds, each a full eleven-fit walk and book; 132 months in
every world, 0 skipped. Gross t median 0.2345,
sd 0.9474, min -1.9381, max 2.1568.
Net-BASE t median -2.1279, sd 0.9572,
min -5.0150, max -0.2151.
Tracked pool `pool_v1.json` sha256 `ab503bfb6a723d75...`; scratch
manifest sha256 `0df2788951ff1067...`.

### The known null
Key 1(a) per world against the median of the other 59: share called significant
**0.0500** against the band [0.01, 0.1] —
`xs_triage.control_verdict` **PASS**. The whole
verdict calls **0** of 60 worlds
SURVIVOR; counts {"NOTHING": 57, "SIGNIFICANT_IMMATERIAL": 3}.

### The plants, eight values
| rho | seed index | share 1(a) | share 1(b) | key 1 | SURVIVOR | median gross Sharpe | median net-BASE Sharpe | median turnover |
|---|---|---|---|---|---|---|---|---|
| 0.005 | 0 | 0.000 | 0.083 | 0 of 12 | 0 of 12 | 0.078 | -0.696 | 2.595 |
| 0.010 | 1 | 0.000 | 0.000 | 0 of 12 | 0 of 12 | 0.151 | -0.622 | 2.613 |
| 0.020 | 2 | 0.667 | 0.667 | 7 of 12 | 0 of 12 | 0.724 | -0.075 | 2.648 |
| 0.025 | 5 | 0.917 | 0.917 | 11 of 12 | 4 of 12 | 1.071 | 0.304 | 2.686 |
| 0.030 | 6 | 1.000 | 1.000 | 12 of 12 | 11 of 12 | 1.581 | 0.756 | 2.705 |
| 0.035 | 7 | 1.000 | 1.000 | 12 of 12 | 12 of 12 | 1.855 | 1.053 | 2.717 |
| 0.040 | 3 | 1.000 | 1.000 | 12 of 12 | 12 of 12 | 2.409 | 1.589 | 2.740 |
| 0.080 | 4 | 1.000 | 1.000 | 12 of 12 | 12 of 12 | 6.364 | 5.574 | 2.825 |

### The two measured floors
| floor | rule | rho | median gross Sharpe | median net-BASE Sharpe |
|---|---|---|---|---|
| key 1 | smallest rho with key 1 passing in >= 10 of 12 | 0.025 | 1.071 | 0.304 |
| whole verdict | smallest rho with SURVIVOR in >= 10 of 12 | 0.030 | 1.581 | 0.756 |

### Determinism
World 1 run twice: bit-identical on every series.

## 9 KNOWN BEFORE THE READ

T-R1 and T-R1b read five of these characteristics singly on this window —
MOM12_1, REV1, LOWVOL, HI52 and MAX — ten ledger rows across the two harness
versions, and every one of the ten is NOTHING. Canon reads 1 and 2 were NIL
(D-CANON-READ1, D-CANON-READ2). The charter expects "real but immaterial after
costs". In the 60 null worlds the gross t has a median of
0.23, the net Sharpe is negative in every world, and
3 worlds were called
SIGNIFICANT_IMMATERIAL, the rate a 5 per cent test produces.

## 10 MDE AGAINST THE BAR

The repo's own rule, quoted from `registrations/r6/registration_v2.json`,
`interpretation_rulings[1]`:

> **RULE-MDE-VS-BAR** — "with tens of thousands of events the MDE sits far below the bar; the verdict is the bar; significance alone promotes nothing"

Applied here it runs the other way. T-R6 had tens of thousands of events and an
MDE far BELOW its bar, so the verdict was the bar. This read has
132 months and an MDE far ABOVE any plausible effect, so the
verdict will be the MDE. The numbers:

| quantity | annualised gross Sharpe |
|---|---|
| key 1 floor (measured, >= 10 of 12 at rho 0.025) | 1.071 |
| whole-verdict floor (measured, >= 10 of 12 at rho 0.030) | 1.581 |
| analytic, 80 per cent power on 132 months, one-sided alpha 0.05 | 0.7497 |
| key 2 bar, on net BASE | 0.50 |

What will decide the verdict is **the whole-verdict floor**, gross Sharpe
1.581. The two floors are not the same bar and the band between them
matters: at rho 0.025 key 1 passed 11 of 12 while only
4 of 12 were SURVIVOR, because base costs on a turnover near 2.7 a month
take the net Sharpe below 0.5. So a gross edge below the key 1 floor
reads NOTHING; one between the two floors clears key 1 and fails key 2, which is
SIGNIFICANT_IMMATERIAL; only one at or above the whole-verdict floor can read
SURVIVOR. The analytic 0.75 is near this instrument's
fifty-per-cent point, not its eighty, because key 1 is two tests and not one.

In Sharpe units these floors carry to the real world. In basis points they do
not: a basis-point floor depends on the real book's monthly volatility, which is
not known before the read.

## 11 WHAT EACH VERDICT WILL MEAN

**NOTHING** — no information before costs at the 5 per cent level on
132 months. It rules out an edge at or above the key 1 floor
and not a smaller one, which at this turnover would not clear the net bar in any
case.

**SIGNIFICANT_IMMATERIAL** — information before costs, at a level the null
reaches one time in twenty, and a net Sharpe under 0.5 at
base costs: an edge this book cannot keep.

**SURVIVOR** — a candidate, not a finding. It earns one confirmation on half B
under a new registration, after a half-B guard is built; the holdout stays
sealed; costs are provisional, so no money test follows without a certified
equity cost row (D-P01-CLOSE).

## 12 LIMITS

- Costs are provisional constants in the research harness — 5 bps a side base,
  15 stress, 50/100 bps a year borrow, commission 0.0 — and **no certified
  equity cost row exists**. A SURVIVOR is therefore a research finding and no
  money test follows without one (D-P01-CLOSE's no-row-no-money-test gate).
- The delisting month's return is imputed 0.0: a name with a month-end close in
  m-1 and none in m is given exactly zero for m and flagged. `bulk_actions` is
  never read, so no liquidation or bankruptcy value enters.
- Sector and category are current-vintage attributes, not point-in-time. Sector
  is a control, and category decides admission to the panel.
- No market capitalisation or shares-outstanding series exists in the snapshot;
  the universe is the top 1,000 by median 21-session dollar volume.
- The position is taken at the formation month-end close on features that stop a
  session earlier, except BETA and LOGDV, which run through s_t.
- The vendor's adjustment semantics are as the repo assumes: returns from
  `closeadj`, the price filter and dollar volume from `closeunadj`.
- Survivorship is evidenced by a 10-name sample of the 8,423 names delisted
  before 2015 (all PRESENT_AT_DEATH), not exhaustively verified.
- 132 scored months. The MDE in section 10 is what that buys.
- Half B has no guard in this repository yet. One must be built before any
  confirmation, and the holdout (2021-01-01 onward) stays sealed.

## 13 The pins

Verified by the gate before the read; any mismatch refuses it.

| artifact | sha256 |
|---|---|
| `banked_panel` | `7f13b1094b059eaa6b69b9ddff9b7b9a2a069d9401b56335673a4314687c30f7` |
| `canon_r3_features.py` | `3329eff5e10307f96dc4ae00cf6e884ca33c8a5d31521076f09ad7d0c8b92220` |
| `canon_r3_gate.py` | `c0ce66094370e99a977b6e79f323afe30ec5a9a5b4b4667ab99c8ec603c0bb93` |
| `canon_r3_machine.py` | `4d1cc4a3204db43dfe66c6866340d9f20008c57faa2926e214a9d2b38c67095d` |
| `canon_r3_mutants.py` | `c22511004ed15663b2cba42168bf7f522223aea0e7c9dc9c620a61848bc49da7` |
| `canon_r3_paths.py` | `4d17bd9a96020d0eaa69f27c94793e6da9e1aa0ddd020252bc0026172a3f790c` |
| `canon_r3_rehearsal.py` | `e6c2e0577748818922b856f2b1c7e1c43fbf888eb7cc4cc6300e26985cb64aa9` |
| `canon_r3_tripwire.py` | `1758c38aecc9df85c62441f8dec204d315644de081b8c7918cf3f3d0e5032cb7` |
| `canon_r3_world.py` | `2ad88dd35577d387cb689dce7a5c860e8854d4d7a6dc3fa491a960b4a29e9238` |
| `controls_v2` | `57eff555b496cd5a487ee8de19e18a8f7b9ec04f0de8c9e88bfde8d3c32b2f6f` |
| `feature_file` | `23b7fbaf9968b2e3130e8aa495e23deb813a49292c9994ad156dccb2f1af4528` |
| `pool_v1` | `ab503bfb6a723d7508b3538297831717fff859331a72c6941368111a18859b4d` |
| `requirements_lock` | `6badb3b4490e577e3dec649644ef948da9dda0f9751be6282136919d783512e9` |
| `run_canon_r3.py` | `d82fbfdc3630b85c81a8c881818a80e102e46edda34863f74c27a4dcdebbf674` |
| `xs_triage.py` | `72faead7f18c091f7ef085980fc5a97af5e39fff6c4cd1e90ba819f5f3ac5b3c` |
