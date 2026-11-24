# Canon read 3, the confirmation — NOT_CONFIRMED

Taken once under seal `0ad6336cbe07d052...` at 2026-11-24T09:00:14Z.
Statistics only; no vendor value appears in this file.

- Verdict: **NOT_CONFIRMED**  (failed: key2)
- Window: top1000_tight_v2_halfB_96mo, 96 scored months, vintage 20261004p
- Pins verified: 22

## Key 1(a)

| | value |
|---|---|
| null centre (pool median gross mean) | +0.0000727081 (+0.7271 bps) |
| observed gross mean | +0.0028495142 (+28.4951 bps) |
| p, one-sided | **0.0037** |
| key 1(a) | **PASS** |

## Key 1(b)

| | value |
|---|---|
| observed gross t | +2.407389 |
| worlds equalling or exceeding it | **1** of 60 |
| allowed | at most 2 |
| key 1(b) | **PASS** |

## Key 2

| | value |
|---|---|
| net BASE annualised Sharpe | +0.4048 |
| bar | 0.50 |
| key 2 | **FAIL** |

## VERDICT: NOT_CONFIRMED

The keys that failed are named. The candidate is closed; half A stands in the record as a discovery that did not replicate; nothing is re-specified and re-read on half B.

## Reported, not voted

| series | n | mean (bps) | sd | t | annualised Sharpe | ratio to read 3 |
|---|---|---|---|---|---|---|
| gross | 96 | +28.4951 | 0.011597 | +2.4074 | +0.8511 | +0.465 |
| net BASE | 96 | +13.5470 | 0.011594 | +1.1448 | +0.4048 | +0.287 |
| net STRESS | 96 | -12.1826 | 0.011588 | -1.0301 | -0.3642 | -0.536 |

- Turnover: 2.1563
- Break-even cost, borrow ignored: **+13.2149 bps a side**
- Bootstrap p of net BASE against zero: 0.1060
- Months scored 96; skipped rank 0; skipped small 0
- Mean cross-section 986.8 of 1,000 (min 977)

| year | months | gross (bps) | net BASE (bps) |
|---|---|---|---|
| 2012 | 1 | +254.900 | +240.734 |
| 2013 | 12 | +9.497 | -5.309 |
| 2014 | 12 | +38.602 | +23.983 |
| 2015 | 12 | +20.344 | +5.634 |
| 2016 | 12 | +24.617 | +9.473 |
| 2017 | 12 | -9.889 | -24.726 |
| 2018 | 12 | +45.303 | +30.537 |
| 2019 | 12 | +42.901 | +27.802 |
| 2020 | 11 | +38.559 | +22.821 |

### The stopped-stock bound, reported and never voted

{
  "months": 96,
  "months_with_a_stopped_position": 81,
  "stopped_positions_total": 163,
  "stopped_positions_max_in_a_month": 6,
  "weight_pos_mean": 0.001999320671961755,
  "weight_pos_max": 0.015582117887072318,
  "weight_pos_sum": 0.1919347845083285,
  "weight_neg_mean": -0.0025627472036416485,
  "weight_neg_min": -0.014902086132633673,
  "weight_neg_sum": -0.24602373154959825
}

{
  "L_0.30": {
    "L": 0.3,
    "mean": 0.000754903510631554,
    "mean_bps": 7.54903510631554,
    "sd": 0.011608096772861976,
    "sharpe": 0.22527917552906337,
    "total_charge_bps": 575.8043535249855
  },
  "L_1.00": {
    "L": 1.0,
    "mean": -0.0006446209597416745,
    "mean_bps": -6.446209597416745,
    "sd": 0.011927540430257405,
    "sharpe": -0.18721651130421646,
    "total_charge_bps": 1919.347845083285
  }
}

## What this does not establish

The 2021 holdout stays sealed. Costs are provisional and no certified equity cost
row exists, so nothing here is a money test (D-P01-CLOSE). The limits of the
registration apply unchanged.
