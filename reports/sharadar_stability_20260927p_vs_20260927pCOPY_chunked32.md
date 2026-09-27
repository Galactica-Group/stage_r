# Sharadar snapshot stability: 20260927p vs 20260927pCOPY

Statistics only. No price, volume or bar-date value appears in this file.

| field | value |
| --- | --- |
| tool version | 1.1.0 |
| fetched_at_utc 20260927p | 2026-09-27T06:53:11Z .. 2026-09-27T07:01:08Z |
| manifest entries 20260927p | 7 |
| manifest sha 20260927p | ffa889a0c0974a2a607ab7b59d3b6141a7801cff0979992e3f0af7447337800c |
| fetched_at_utc 20260927pCOPY | 2026-09-27T06:53:11Z .. 2026-09-27T07:01:08Z |
| manifest entries 20260927pCOPY | 7 |
| manifest sha 20260927pCOPY | ffa889a0c0974a2a607ab7b59d3b6141a7801cff0979992e3f0af7447337800c |
| tables only in A | none |
| tables only in B | none |
| entries only in A | none |
| entries only in B | none |
| entries unmapped to a table | none |

## Per-table row membership

| table | key fields | key source | only A | only B | in both | differing (vendor fields) | watermark-only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actions | (whole row) | none (whole-row set comparison) | 0 | 0 | 709246 | None | None |
| sp500 | ticker,date,action | sprint brief (vendor declares: none) | 0 | 0 | 59682 | 0 | 0 |
| stocks | ticker,date | sprint brief (vendor declares: none) | 0 | 0 | 45372755 | 0 | 0 |
| tickers | permaticker | sprint brief (vendor declares: none) | 0 | 0 | 43958 | 0 | 0 |

## Intra-snapshot key collapse

Several requests overlap, so a table is the concatenation of its source entries
deduplicated on the key. A conflict is one key carrying two different value sets
inside a single snapshot — a narrowing artefact where the key used is narrower
than the vendor's declared key, not a between-snapshot change.

| table | rows before dedupe (A / B) | rows after (A / B) | key conflicts (A / B) | source entries (A / B) |
| --- | --- | --- | --- | --- |
| sp500 | 59687 / 59687 | 59682 / 59682 | 0 / 0 | 2 / 2 |
| stocks | 45372760 / 45372760 | 45372755 / 45372755 | 5 / 5 | 2 / 2 |
| tickers | 74288 / 74288 | 43958 / 43958 | 18714 / 18714 | 2 / 2 |

`lastupdated` is reported as a watermark count only; it never enters classification.

## Per-field changed counts

- **sp500**: no field changed
- **stocks**: no field changed
- **tickers**: no field changed

## Price tables — magnitude of change and classification

### stocks

Tickers compared: 20990. Class tally: UNCHANGED=20990

| field | changed | abs magnitude | rel magnitude | <=7 | 8-30 | 31-365 | >365 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open | 0 | n/a | n/a | 0 | 0 | 0 | 0 |
| high | 0 | n/a | n/a | 0 | 0 | 0 | 0 |
| low | 0 | n/a | n/a | 0 | 0 | 0 | 0 |
| close | 0 | n/a | n/a | 0 | 0 | 0 | 0 |
| volume | 0 | n/a | n/a | 0 | 0 | 0 | 0 |
| closeadj | 0 | n/a | n/a | 0 | 0 | 0 | 0 |
| closeunadj | 0 | n/a | n/a | 0 | 0 | 0 | 0 |

REVISION tickers: none.

- **funds**: no price analysis (table absent or not keyed by ticker/date).
## Examples file

Path (untracked, gitignored): `data/sharadar/diff_examples_20260927p_vs_20260927pCOPY.csv` — 0 row(s), capped at 10.

