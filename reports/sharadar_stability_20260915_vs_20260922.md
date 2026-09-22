# Sharadar snapshot stability: 20260915 vs 20260922

Statistics only. No price, volume or bar-date value appears in this file.

| field | value |
| --- | --- |
| tool version | 1.0.0 |
| fetched_at_utc 20260915 | 2026-09-15T20:30:07Z .. 2026-09-15T20:36:32Z |
| manifest entries 20260915 | 63 |
| manifest sha 20260915 | ee0c5e84df89e743a056615d5a2cac5b4ecd76c7d516bb9288a40eb783b1de4f |
| fetched_at_utc 20260922 | 2026-09-22T16:10:24Z .. 2026-09-22T17:10:06Z |
| manifest entries 20260922 | 63 |
| manifest sha 20260922 | 10fc17e293c904d4419080e94b4acb0b1fa967b9727d8dd6e6bf9032b7177f3b |
| tables only in A | none |
| tables only in B | none |
| entries only in A | none |
| entries only in B | none |
| entries unmapped to a table | schema_index |

## Per-table row membership

| table | key fields | key source | only A | only B | in both | differing (vendor fields) | watermark-only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actions | action,contraname,contraticker,date,name,ticker | vendor isprimarykey (desc_all) | 5 | 5 | 0 | 0 | 0 |
| daily | date,ticker | vendor isprimarykey (desc_all) | 5 | 5 | 0 | 0 | 0 |
| descriptions | table,indicator | sprint brief (vendor declares: indicator,table) | 0 | 0 | 384 | 25 | 0 |
| events | date,ticker | vendor isprimarykey (desc_all) | 2 | 2 | 3 | 0 | 0 |
| fundamentals | date,dimension,reportperiod,ticker | vendor isprimarykey (desc_all) | 0 | 0 | 5 | 0 | 0 |
| funds | ticker,date | sprint brief (vendor declares: date,ticker) | 5 | 5 | 0 | 0 | 0 |
| holdings | date,investorid,securitytype,ticker | vendor isprimarykey (desc_all) | 5 | 5 | 0 | 0 | 0 |
| holdings_investor | date,investorid | vendor isprimarykey (desc_all) | 0 | 0 | 4 | 0 | 0 |
| holdings_ticker | date,ticker | vendor isprimarykey (desc_all) | 0 | 0 | 5 | 5 | 0 |
| insiders | date,formtype,ownername,rownum,ticker | vendor isprimarykey (desc_all) | 5 | 5 | 0 | 0 | 0 |
| metrics | ticker,date | sprint brief (vendor declares: date,ticker) | 5 | 5 | 0 | 0 | 0 |
| sp500 | ticker,date,action | sprint brief (vendor declares: action,date,ticker) | 30 | 30 | 120 | 0 | 0 |
| stocks | ticker,date | sprint brief (vendor declares: date,ticker) | 155 | 5 | 37440 | 3744 | 3744 |
| tickers | permaticker | sprint brief (vendor declares: permaticker,table,ticker) | 5 | 20 | 20971 | 14678 | 6430 |

## Intra-snapshot key collapse

Several requests overlap, so a table is the concatenation of its source entries
deduplicated on the key. A conflict is one key carrying two different value sets
inside a single snapshot — a narrowing artefact where the key used is narrower
than the vendor's declared key, not a between-snapshot change.

| table | rows before dedupe (A / B) | rows after (A / B) | key conflicts (A / B) | source entries (A / B) |
| --- | --- | --- | --- | --- |
| actions | 5 / 5 | 5 / 5 | 0 / 0 | 1 / 1 |
| daily | 5 / 5 | 5 / 5 | 0 / 0 | 1 / 1 |
| descriptions | 389 / 389 | 384 / 384 | 0 / 0 | 2 / 2 |
| events | 5 / 5 | 5 / 5 | 0 / 0 | 1 / 1 |
| fundamentals | 5 / 5 | 5 / 5 | 0 / 0 | 1 / 1 |
| funds | 5 / 5 | 5 / 5 | 0 / 0 | 1 / 1 |
| holdings | 5 / 5 | 5 / 5 | 0 / 0 | 1 / 1 |
| holdings_investor | 4 / 4 | 4 / 4 | 0 / 0 | 1 / 1 |
| holdings_ticker | 5 / 5 | 5 / 5 | 0 / 0 | 1 / 1 |
| insiders | 5 / 5 | 5 / 5 | 0 / 0 | 1 / 1 |
| metrics | 5 / 5 | 5 / 5 | 0 / 0 | 1 / 1 |
| sp500 | 155 / 155 | 150 / 150 | 0 / 0 | 2 / 2 |
| stocks | 90527 / 90227 | 37595 / 37445 | 0 / 0 | 10 / 10 |
| tickers | 20982 / 20997 | 20976 / 20991 | 2 / 2 | 6 / 6 |

`lastupdated` is reported as a watermark count only; it never enters classification.

## Per-field changed counts

- **actions**: no field changed
- **daily**: no field changed
- **descriptions**: description=25, title=1
- **events**: no field changed
- **fundamentals**: no field changed
- **funds**: no field changed
- **holdings**: no field changed
- **holdings_investor**: no field changed
- **holdings_ticker**: shrholders=5, shrunits=5, shrvalue=5, totalvalue=5
- **insiders**: no field changed
- **metrics**: no field changed
- **sp500**: no field changed
- **stocks**: closeadj=3744, lastupdated=3744
- **tickers**: category=7, companysite=8, cusips=56, exchange=2, figi=52, firstpricedate=7872, firstquarter=26, industry=17, isdelisted=9, lastpricedate=6347, lastquarter=80, lastupdated=6430, name=15, relatedtickers=3264, scalemarketcap=11, scalerevenue=1, ticker=18

## Price tables — magnitude of change and classification

### stocks

Tickers compared: 30. Class tally: ADJUSTMENT=3, UNCHANGED=27

| field | changed | abs magnitude | rel magnitude | <=7 | 8-30 | 31-365 | >365 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open | 0 | n/a | n/a | 0 | 0 | 0 | 0 |
| high | 0 | n/a | n/a | 0 | 0 | 0 | 0 |
| low | 0 | n/a | n/a | 0 | 0 | 0 | 0 |
| close | 0 | n/a | n/a | 0 | 0 | 0 | 0 |
| volume | 0 | n/a | n/a | 0 | 0 | 0 | 0 |
| closeadj | 3744 | n=3744 min=0.228 med=0.425 p95=0.688 max=0.919 | n=3744 min=0.00180208 med=0.00587662 p95=0.00594652 max=0.00595806 | 12 | 45 | 693 | 2994 |
| closeunadj | 0 | n/a | n/a | 0 | 0 | 0 | 0 |

REVISION tickers: none.

### funds

Tickers compared: 0. Class tally: none

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

## Examples file

Path (untracked, gitignored): `data/sharadar/diff_examples_20260915_vs_20260922.csv` — 10 row(s), capped at 10.

