# Canon read 3 — the machine on the stock panel — SURVIVOR

The third and last registered read of the Canon campaign (D-CANON-CHARTER part
3), taken once, under seal `491b278c96134bbd...`, at 2026-11-21T23:29:48Z.
Statistics only; no vendor price appears in this file.

- Verdict: **SURVIVOR**
- Window: top1000_tight_v2_halfA_132mo, 132 scored months, vintage 20261004p
- Pins verified: 15 artifacts.

## Key 1 — information, on GROSS, both parts

(a) stationary bootstrap of months, mean block 3.0,
R 10000, seed 20261121, on the gross series less the
median gross mean of the 60 worlds.

| | value |
|---|---|
| null centre (pool median gross mean) | +0.00016924 (+1.6924 bps) |
| observed gross mean | +0.00657669 (+65.7669 bps) |
| p, one-sided | **0.0001** |
| level | p <= 0.05 |
| **key 1(a)** | **PASS** |

(b) the gross t against the 60 worlds' gross t.

| | value |
|---|---|
| observed gross t | +6.0708 |
| worlds equalling or exceeding it | **0** of 60 |
| allowed | at most 2 |
| p = (1 + count) / 61 | 0.0164 |
| **key 1(b)** | **PASS** |

**Key 1 = PASS**

## Key 2 — materiality, on NET BASE

| | value |
|---|---|
| net BASE annualised Sharpe | +1.4083 |
| bar | 0.50 |
| **key 2** | **PASS** |

## VERDICT: SURVIVOR

A candidate, not a finding. It earns one confirmation on half B under a new registration, after a half-B guard is built; the holdout stays sealed; costs are provisional, so no money test follows without a certified equity cost row (D-P01-CLOSE).

## Reported, not voted

| series | n | mean (bps) | sd | t | annualised Sharpe |
|---|---|---|---|---|---|
| gross | 132 | +65.7669 | 0.012446 | +6.0708 | +1.8304 |
| net BASE | 132 | +50.6008 | 0.012447 | +4.6707 | +1.4083 |
| net STRESS | 132 | +24.4351 | 0.012449 | +2.2552 | +0.6800 |

- Mean monthly traded notional (turnover): 2.1999
- Break-even cost, borrow ignored: **+29.8954 bps a side**
  (base is 5.0, stress 15.0)
- Bootstrap p of the net BASE series against zero: 0.0001
- Months scored 132; skipped rank-deficient
  0; skipped too small
  0
- Mean cross-section 990.3 of 1,000 (min 983)

| year | months | gross mean (bps) | net BASE mean (bps) |
|---|---|---|---|
| 2001 | 1 | +41.711 | +27.544 |
| 2002 | 12 | +100.350 | +84.470 |
| 2003 | 12 | +93.800 | +77.817 |
| 2004 | 12 | +121.931 | +106.527 |
| 2005 | 12 | +88.876 | +73.783 |
| 2006 | 12 | +75.913 | +61.257 |
| 2007 | 12 | +7.294 | -7.772 |
| 2008 | 12 | +64.289 | +49.591 |
| 2009 | 12 | -12.777 | -27.955 |
| 2010 | 12 | +46.410 | +31.453 |
| 2011 | 12 | +63.819 | +48.732 |
| 2012 | 11 | +76.425 | +61.538 |

## What this does not establish

Half B is unopened and the holdout sealed. Costs are provisional constants and no
certified equity cost row exists, so nothing here is a money test (D-P01-CLOSE).
The limits of the sealed registration apply unchanged.
