# Sprint R0-D verdict: snapshot 20261004p against 20260927p

Statistics only. No price, volume, dividend amount or other vendor value appears
in this file; per-row detail stays in the untracked snapshot tree.

Diff: `sharadar_snapshot_diff.py` v1.2.0, 32 ticker partitions, wall 2221.07 s, peak RSS 2.600 GiB.
Read parameters, fixed before the pull: `action_window_end`=2026-10-02, `action_window_start`=2026-09-25, `onlyb_cutoff_date`=2026-09-25, `revision_cutoff_date`=2026-09-01.

## Verdict by pre-registered read

| read | expectation | measured | verdict |
| --- | --- | --- | --- |
| R1 | stocks only-A == 0 | **22,545** only-A rows, every year 1999-2026 | **FINDING** |
| R2 | only-B dated after 2026-09-25 only | 43,896 only-B: 31,564 in the five new sessions, **12,332 BACKFILL** | **FINDING** |
| R3a | REVISION rows dated before 2026-09-01 == 0 | **120,424** (tool rule) / **121,395** (brief gloss) | **FINDING** |
| R3b | every ADJUSTMENT ticker explained; unexplained == 0 | 312 of 312 explained by a dividend in window; **unexplained 0** | **PASS** |
| R4 | value change without watermark advance == 0 | **5** as emitted; all 5 proven instrument artefact, 0 vendor | **FINDING (artefact)** |
| R5 | delisted records unchanged; only-A == 0 | only-A **0**; **2,470 of 18,263** delisted records moved a field | **FINDING** |
| R6 | only-A by type an expected withdrawal class; sp500 set size beside 503 | actions only-A 2,923 across 24 types; sp500 current **503**, **0 adds/removes in the week** | **FINDING (actions)** / **PASS (sp500)** |
| R7 | rows per year identical 1997-2025; 2026 grows by 2026-dated only-B | **26 of 30 years differ**; 2026 +31,629 vs 32,400 only-B | **FINDING** |

No read was softened to fit. Each finding is reported with its counts and dates
and left for ruling.

## R1 and R2 — the counts, and what the probe found behind them

| quantity | value |
| --- | --- |
| only-A rows (vanished from the key space) | 22545 |
| only-B rows | 43896 |
| only-B in the five new sessions (after 2026-09-25) | 31564 |
| only-B BACKFILL (dated at or before 2026-09-25) | 12332 |
| BACKFILL on names new to the key space | 11980 rows over 8 tickers |
| BACKFILL on names already present | 352 rows over 54 tickers |

New sessions by date: 2026-09-28=6329, 2026-09-29=6326, 2026-09-30=6322, 2026-10-01=6320, 2026-10-02=6267.

Only-A rows by year: 1999=371, 2000=404, 2001=260, 2002=443, 2003=504, 2004=298, 2005=252, 2006=438, 2007=810, 2008=563, 2009=504, 2010=619, 2011=785, 2012=1303, 2013=1503, 2014=1204, 2015=1009, 2016=1008, 2017=1004, 2018=1200, 2019=1468, 2020=1251, 2021=1008, 2022=955, 2023=750, 2024=851, 2025=1009, 2026=771.

BACKFILL rows by year: 1997=1, 1998=252, 2007=271, 2008=253, 2009=66, 2011=29, 2012=250, 2013=252, 2014=443, 2015=637, 2016=756, 2017=753, 2018=949, 2019=1008, 2020=1012, 2021=1008, 2022=955, 2023=750, 2024=851, 2025=1000, 2026=836.

A read-only probe (scratch, untracked) measured the cause rather than inferring it:

| probe measurement | value |
| --- | --- |
| distinct tickers, P1 / P2 | 20990 / 20997 |
| tickers absent from P2 entirely | 6 |
| tickers new in P2 entirely | 13 |
| tickers holding fewer rows in P2 | 23, carrying 22520 of the 22,545 only-A rows |
| of the 13 new tickers, how many carry a pre-2026 first price date | 6 |
| permatickers that disappeared (R5 only-A) | 0 |

The 13 tickers new to P2 include six whose first price date lies years in the past
(earliest 2007-01-25), and the 23 shrinking tickers lose whole histories rather than
recent bars. No permaticker disappeared. The pattern is **symbol migration**: the
vendor's stable identity is `permaticker`, and a company's history moves between
ticker strings, which a `(ticker, date)` key reads as deletion plus creation.

## R3 — value change, classified

| quantity | tool rule (closeunadj, volume) | brief gloss (OHLC, volume, closeunadj) |
| --- | --- | --- |
| rows whose value moved | 1425280 | 1425280 |
| REVISION rows | 121890 | 123264 |
| ADJUSTMENT-only rows | 1303390 | 1302016 |
| REVISION dated before 2026-09-01 | **120424** | **121395** |
| REVISION dated 2026-09-01 or later | 1466 | 1869 |

The brief glosses R0-B-0's split as `open, high, low, close, volume, closeunadj`;
the tool's actual rule is narrower, treating only `closeunadj` and `volume` as
as-traded. R0-B's banked "0 of 30" came from the tool's rule, so that is the
headline here and the gloss is reported beside it. The two differ by 1374 rows.

Per-ticker classes: ADJUSTMENT=312, REVISION=402, TAIL=1422, UNCHANGED=18847 over 20983 tickers compared.

REVISION rows dated 2026-09-01 or later, by date (tool rule): 2026-09-01=20, 2026-09-02=19, 2026-09-03=19, 2026-09-04=19, 2026-09-08=19, 2026-09-09=19, 2026-09-10=19, 2026-09-11=19, 2026-09-14=19, 2026-09-15=19, 2026-09-16=19, 2026-09-17=19, 2026-09-18=19, 2026-09-21=21, 2026-09-22=20, 2026-09-23=20, 2026-09-24=20, 2026-09-25=1137.

REVISION rows dated before 2026-09-01, by year (tool rule): 1997=10, 1998=2670, 1999=3053, 2000=3834, 2001=4497, 2002=4756, 2003=4756, 2004=5106, 2005=6065, 2006=6134, 2007=6745, 2008=7403, 2009=6850, 2010=6402, 2011=5410, 2012=4526, 2013=3467, 2014=3259, 2015=2405, 2016=1868, 2017=1506, 2018=1691, 2019=1562, 2020=1935, 2021=3088, 2022=4000, 2023=4352, 2024=4805, 2025=5094, 2026=3175.

### Is the pre-cutoff revision count split restatement or genuine revision?

| probe measurement | value |
| --- | --- |
| new action rows in P2 | 8740 |
| of those, split-like (`split`, `adrratiosplit`) | 54 rows over 31 tickers |
| REVISION tickers | 402 |
| REVISION tickers carrying a new split-like action | **17** |
| REVISION tickers whose `closeunadj` moved | 335 |
| of those, moving by a single constant ratio | 245 |

Only 17 of 402 REVISION tickers have a new split, so the pre-cutoff count is **not**
mostly split restatement. The remainder are per-bar corrections to as-traded data,
often a single bar per ticker. R0-B's "0 of 30" was measured on a 30-name, five-year
large-cap slice and was never evidence about the full 28-year, 20,997-name universe.

### ADJUSTMENT explanation (R3b)

| quantity | value |
| --- | --- |
| window | (2026-09-25, 2026-10-02] |
| ADJUSTMENT tickers | 312 |
| corporate actions in the window | 1729 |
| of those, on an ADJUSTMENT ticker | 318 |
| explained | **312** |
| unexplained | **0** |
| explained by type | dividend=318 |

Every one of the 312 ADJUSTMENT tickers is backed by a dividend inside the window,
and the 1395726 adjusted-series row changes carry a median relative move of about 0.9 per
cent, the scale a dividend re-adjustment produces. R3b passes outright.

## R4 — the watermark, and why its 5 are not vendor behaviour

| quantity | value |
| --- | --- |
| rows whose value moved | 1425280 |
| of those, watermark advanced | 1425275 |
| of those, watermark did **not** advance | **5** |
| watermark moved with no value change (benign) | 15005 |
| watermark unparseable | 0 |

All five were located. They are five bars of one ticker, dated 2026-09-21 to
2026-09-25, differing on `volume` alone, where the JSON tier sample renders the
integer without a decimal point and the CSV bulk render carries a trailing `.0`.
The numeric values are equal and the watermark correctly did not move, because
nothing changed. A same-snapshot control reproduces the identical difference
between the tier and bulk renders of snapshot P1 alone, which is what proves it
an instrument artefact rather than a vendor one.

**On the vendor's own behaviour the watermark is reliable: 1425275 of 1425275 genuine value
changes advanced it, and 0 did not.** The finding is against the instrument.

### Scope of the same artefact

The snapshot pair mixes two request shapes in three logical tables, because P1
carried three small tier probes alongside the bulk pull and P2 carried one. Where
a tier row wins the key dedupe on one side and a bulk row wins on the other, a
formatting difference reads as a change. Measured reach:

| table | tier rows | artefact |
| --- | --- | --- |
| stocks | 5 in each snapshot | the 5 R4 rows; inflates `volume` changed by 5 and tool-rule REVISION by 5, all dated after the settle cutoff |
| tickers | 5 in P1 only | 5 spurious `table` changes: the tier render labels these rows `holdings_investor` where bulk labels them `SF3B` |
| sp500 | 5 in P1 only | 5 keys collapsed in P1's dedupe; no field change survived |

The pre-cutoff REVISION count of 120424 is untouched by this artefact. Recommended for
the next snapshot: pull the tier control under its own label, so the diffed snapshot
holds one render per table.

## R5 — the master, delisted against active

| quantity | value |
| --- | --- |
| permatickers only in P1 | **0** |
| permatickers only in P2 (new listings) | 34 |
| permatickers in both | 43958 |
| delisted in P1 | 18263, of which **2470** moved a field |
| active in P1 | 25695, of which 12718 moved a field |

Delisted records, changed fields: relatedtickers=2386, exchange=44, lastupdated=41, figi=26, lastpricedate=14, firstquarter=11, scalemarketcap=8, firstpricedate=7, lastquarter=6, ticker=3, siccode=1, sicindustry=1, table=1.

Active records, changed fields: lastpricedate=12667, lastupdated=7312, relatedtickers=1242, lastquarter=148, cusips=79, figi=46, companysite=10, scalemarketcap=8, firstquarter=7, table=5, ticker=5, category=4, name=4, scalerevenue=4, firstpricedate=3, currency=1, location=1.

No permaticker disappeared, which is the half of R5 that passes and the same fact
that makes `permaticker` the stable identity behind R1. The delisted half fails:
2470 closed records moved a field. 2386 of those 2470 are `relatedtickers`, a
cross-reference that moves when other names move; the substantive remainder is
small but real, and includes `ticker`, `firstpricedate` and `lastpricedate`.

**Load-bearing limitation.** This table is compared on `permaticker` alone, which is
narrower than its real key: the master covers five vendor sub-tables and one
permaticker appears once per sub-table. 30330 of 74288 rows collapse in the dedupe and
18,714 permatickers carry conflicting value sets *within* a single snapshot. Some of the
field changes above may therefore record which sub-table's row happened to be first,
not a change between snapshots. R5's counts should be read as an upper bound until
the comparison is re-keyed on `(table, permaticker)`.

## R6 — event tables by type

### actions

Compared as whole rows, because this table declares no key in the snapshot; an
edited row therefore appears once on each side rather than as a change.

| quantity | value |
| --- | --- |
| only-A rows | 2923 across 24 types |
| only-B rows | 9593 across 25 types |
| only-A date span | 1998-04-16 .. 2026-09-25 |

Only-A by type: acquisitionby=14, acquisitioncash=9, acquisitionof=14, acquisitionstock=3, bankruptcyliquidation=8, delisted=24, dividend=394, exchangefrom=192, exchangeto=187, listed=33, namechangefrom=7, namechangeto=7, regulatorydelisting=3, relation=1976, sicchangefrom=5, sicchangeto=5, spacmerger=1, spacunitseparation=1, spinoff=1, spinoffdividend=2, split=12, spunofffrom=1, tickerchangefrom=12, tickerchangeto=12.

Only-B by type: acquisitionby=15, acquisitioncash=11, acquisitionof=15, acquisitionstock=3, adrratiosplit=4, bankruptcyliquidation=7, delisted=38, dividend=2149, exchangefrom=2589, exchangeto=2584, initiated=1, listed=64, namechangefrom=9, namechangeto=9, regulatorydelisting=8, relation=1978, sicchangefrom=5, sicchangeto=5, spacunitseparation=1, spinoff=3, spinoffdividend=3, split=53, spunofffrom=3, tickerchangefrom=18, tickerchangeto=18.

R6 anticipated an "announced-not-effective, since withdrawn" class. This vendor's
taxonomy has no `announcement` type, so that class does not exist as named. What the
counts show instead is near-symmetry on the churning types (`relation` 1,976 against
1,978, `exchangefrom` 192 against 2,589), which is the signature of rows being
**edited in place** rather than withdrawn, spread across the whole history
(1998-04-16 to 2026-09-25). Distinguishing edit from withdrawal needs a declared key
for this table; it cannot be settled from a whole-row comparison.

The table also carries rows dated beyond the run date (2026-10-05 and 2026-10-06),
i.e. declared future effective dates, consistent with its own span.

### sp500

| quantity | P1 | P2 |
| --- | --- | --- |
| `current` rows | 503 | 503 |
| `current` stamp date | 2026-09-27 | 2026-10-03 |
| `historical` rows | 57194 | 57697 |
| `added` rows | 1241 | 1241 |
| `removed` rows | 744 | 744 |

**Adds and removes in the week: 0.** The current set is 503, unchanged from P1, and
the `added`/`removed` totals are identical across the two snapshots.

The 509 only-A and 1,012 only-B rows are fully accounted for and benign: the vendor
archives the prior week's `current` block as `historical` (+503, matching the
historical row growth exactly) and writes a new `current` block under the new stamp
date. The residue is 3 `added` and 3 `removed` historical events whose dates moved
forward by three or four days (2001-07-06 to 2001-07-09, 2017-01-13 to 2017-01-17,
2024-07-05 to 2024-07-08) — a re-dating of settled membership events, worth a ruling
of its own since it means historical membership dates are not themselves stable.

## R7 — universe statistics on P2, beside P1

| statistic | P1 (20260927p) | P2 (20261004p) |
| --- | --- | --- |
| price rows | 45372755 | 45394106 |
| price span | 1997-12-31 .. 2026-09-25 | 1997-12-31 .. 2026-10-02 |
| distinct tickers in prices | 20990 | 20997 |
| SEP master rows | 20991 | 20997 |
| SEP active | **6339** | **6332** |
| SEP delisted | **14652** | **14665** |
| survivorship: price rows on delisted names | 55.5348% | 55.5321% |
| master lastpricedate max | 2026-09-25 | 2026-10-02 |
| sp500 reconciliation steps exact | 100.0% | 100.0% |
| actions rows | 709246 | 715916 |
| delisted presence check | 10/10 present at death | 10/10 present at death |

Rows per year are identical on only 4 of 30 years (2021-2024); 26 differ. 2026 grows
by 31,629, not by the 32,400 only-B rows dated 2026, so R7 fails as written. It fails
for exactly one reason: the expectation was drafted assuming R1 held. Every year
reconciles on the identity

    rows_per_year(P2) - rows_per_year(P1)  ==  onlyB(year) - onlyA(year)

which holds on **30 of 30 years with no exception**, and at table level
`rows_in_both + onlyA == P1 rows` and `rows_in_both + onlyB == P2 rows` both hold
exactly. The diff and the universe statistics are mutually consistent; the per-year
movement is the symbol migration of R1 seen from the other side.

The S&P 500 reconciliation remains exact at 100 per cent of steps, over 116 snapshot
dates against 115 in P1, with 0 extras and 0 missing.

## What the free-tier and paid-tier picture now is

Full history is a fixed vintage in the sense that matters: the start edge did not
recede (1997-12-31 on both), no permaticker was lost, and the S&P 500 membership
chain still reconciles exactly. What is **not** stable across a week is: the
`(ticker, date)` key, as-traded price and volume on settled bars, the `table` labels
and cross-references of closed records, historical corporate-action rows, and the
dates of three settled membership events. A point-in-time universe built on this
data must key on `permaticker`, and any banked result computed from as-traded prices
should record the snapshot label it was computed against.

