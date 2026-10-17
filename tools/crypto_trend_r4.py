"""
================================================================================
Script:       crypto_trend_r4.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      The T-R4 engine: a point-in-time top-20 Binance spot basket, the
              ten-month trend rule long-or-USDT against buy-and-hold, with the
              random-timing null and both registered controls.
Entry Point:  No — imported by tools/run_t_r4.py
Dependencies: stdlib, numpy, pandas, pyarrow. Imports nothing from any other
              repository; in particular it imports nothing from the factory tree.
Last Sprint:  Sprint R4-1 — T-R4 crypto trend: seal, controls, half-A read (2026-10-17)
================================================================================

Every parameter comes from the sealed registration JSON, whose sha256 must match
the one passed in; load_registration refuses otherwise. Nothing is hardcoded
that the registration fixes, and the few integers the registration states in
prose rather than in a numeric field (the ten-month lookback, the top 20, the
ninety-bar window, the MDE z) are declared here and pinned against that prose by
a test, so the code and the seal cannot drift apart silently.

A SLEEVE, not a symbol, is the unit. A symbol's daily history is cut at every
interior gap of one or more missing calendar days; each piece is its own sleeve
with its own eligibility clock, and a sleeve is sold at its last close in both
books when it ends. Because a sleeve's bars are contiguous its month-end closes
are consecutive calendar months, so a return can only ever be formed inside one
sleeve — the structural reason no return crosses a gap, asserted rather than
assumed.

Cash is USDT at zero yield, so monthly returns ARE excess returns and the two
books are compared on raw Sharpe. That has one useful consequence the null
exploits: dSR differs from the trend book's own Sharpe by a constant (the
buy-and-hold Sharpe), so the null's spread, the MDE and every timing p can be
computed from trend-side Sharpes alone, without ever forming the buy-and-hold
book's performance. Steps 1 to 3 of the registered order therefore touch no
observed-book figure at all.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "1.0.0"

CASH = "USDT"
MONTHS_PER_YEAR = 12

# Stated in the registration as prose rather than as numeric fields; pinned
# against that prose by test_registration_prose_pins_the_constants.
LOOKBACK = 10            # "the mean of its last 10 month-end closes"
MIN_MONTH_ENDS = 10      # "at least 10 month-end closes (t included)"
MIN_BARS = 90            # "at least 90 daily bars ending at t"
RANK_BARS = 90           # "the sum of quote_volume over the 90 daily bars ending at t"
TOP_N = 20               # "the top 20 by that ranking"
Z80 = 0.8416             # "+ 0.8416 * sd(null dSR)"
TOKENIZED_SUFFIX = "B"   # "whose base ends in the letter B"
QUOTE = "USDT"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registration(path, expected_sha):
    """Load the seal and refuse any file whose sha256 is not the one passed in."""
    path = Path(path)
    got = sha256_file(path)
    if got != expected_sha:
        raise AssertionError(
            "registration sha256 mismatch: file %s has %s, --sha said %s"
            % (path, got, expected_sha))
    reg = json.loads(path.read_text())
    reg["_sha256"] = got
    reg["_path"] = str(path)
    return reg


def window_bounds(reg):
    """(first_earning_month, last_earning_month, exclusive_max) from the seal."""
    s = reg["window"]["half_A_earning_months"]
    lo, hi = [x.strip() for x in s.split("..")]
    first = pd.Period(lo, freq="M")
    last = pd.Period(hi, freq="M")
    return first, last, (last + 1).start_time.tz_localize("UTC")


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
def base_asset(symbol):
    if not symbol.endswith(QUOTE):
        raise AssertionError("symbol %r is not %s-quoted" % (symbol, QUOTE))
    return symbol[: -len(QUOTE)]


def resolve_universe(manifest, reg):
    """Apply the registered exclusions to the vintage manifest. Counts and names.

    The tokenized-equity rule is resolved against the whole vintage, which is
    what its 'exactly 95' pin counts, so a symbol may appear in that set AND in a
    named list; the pool is the set difference against the UNION.
    """
    ex = reg["universe"]["exclusions"]
    all_syms = sorted(manifest["symbols"].keys())
    present = set(all_syms)

    pegged = list(ex["pegged_or_metal_backed"])
    wrapped = list(ex["wrapped_or_staked_duplicates"])
    missing = [s for s in pegged + wrapped if s not in present]

    cutoff = "2026-06-11"
    tok = sorted(
        s for s in all_syms
        if base_asset(s).endswith(TOKENIZED_SUFFIX)
        and len(base_asset(s)) > 1
        and manifest["symbols"][s]["first_open_time"][:10] >= cutoff)
    if len(tok) != 95:
        raise AssertionError(
            "tokenized-equity set resolved to %d symbols, registration pins 95: HALT"
            % len(tok))

    union = set(pegged) | set(wrapped) | set(tok)
    pool = [s for s in all_syms if s not in union]
    return {
        "vintage_symbols": len(all_syms),
        "pegged_or_metal_backed": pegged,
        "wrapped_or_staked_duplicates": wrapped,
        "tokenized_equities": tok,
        "tokenized_cutoff": cutoff,
        "registered_names_absent_from_vintage": missing,
        "double_listed": sorted(set(pegged + wrapped) & set(tok)),
        "excluded_union": len(union),
        "pool": pool,
        "pool_size": len(pool),
    }


# ---------------------------------------------------------------------------
# Sleeves
# ---------------------------------------------------------------------------
def sleeve_bounds(index):
    """[(start, stop)) over a sorted daily DatetimeIndex, cut at every gap."""
    if len(index) == 0:
        return []
    d = np.diff(np.asarray(index, dtype="datetime64[D]").astype("int64"))
    cuts = np.flatnonzero(d > 1) + 1
    starts = np.concatenate(([0], cuts))
    stops = np.concatenate((cuts, [len(index)]))
    return list(zip(starts.tolist(), stops.tolist()))


def count_sleeves(vintage_dir, symbols):
    """Sleeves per symbol over the WHOLE vintage, from the open_time index only.

    Presence, not price: only the index column is read.
    """
    out = {}
    for s in symbols:
        idx = pd.read_parquet(Path(vintage_dir) / ("%s.parquet" % s),
                              columns=[]).index
        out[s] = sleeve_bounds(idx)
    return out


def load_vintage(vintage_dir, symbols, exclusive_max):
    """Daily close and quote_volume per symbol, strictly before exclusive_max.

    Filtered at read AND asserted after load, so half B and the holdout cannot
    enter by accident.
    """
    cut = pd.Timestamp(exclusive_max)
    out, spans = {}, {}
    for s in symbols:
        df = pd.read_parquet(Path(vintage_dir) / ("%s.parquet" % s),
                             columns=["close", "quote_volume"])
        raw_max = df.index.max()
        keep = df.index < cut
        df = df.loc[keep]
        if len(df) and not (df.index.max() < cut):
            raise AssertionError("%s loaded a bar at or past %s" % (s, cut))
        if len(df):
            out[s] = df.sort_index()
            spans[s] = (str(df.index[0].date()), str(df.index[-1].date()),
                        int(len(df)), str(raw_max.date()))
    return out, spans


def sleeve_records(daily):
    """One record per sleeve: month-end closes, bar counts, 90-bar quote volume.

    A sleeve's bars are contiguous, so its month periods are consecutive; the
    month-end close is the last bar of the month INSIDE the sleeve, which is the
    registered in-sleeve fallback and, in a sleeve's final month, its last close.
    """
    recs = []
    for sym in sorted(daily):
        df = daily[sym]
        close = df["close"].to_numpy(dtype=float)
        qv = df["quote_volume"].to_numpy(dtype=float)
        per = pd.PeriodIndex(df.index, freq="M")
        cum_qv = np.concatenate(([0.0], np.cumsum(qv)))
        for k, (a, b) in enumerate(sleeve_bounds(df.index)):
            p = per[a:b]
            # last bar position of each month, within the sleeve
            last_pos = {}
            for i in range(a, b):
                last_pos[per[i]] = i
            months = sorted(last_pos)
            if len(months) > 1:
                step = [(months[i + 1] - months[i]).n for i in range(len(months) - 1)]
                if any(x != 1 for x in step):
                    raise AssertionError(
                        "%s sleeve %d has non-consecutive months: a sleeve's bars "
                        "must be contiguous" % (sym, k))
            rows = []
            for mp in months:
                i = last_pos[mp]
                n_bars = i - a + 1
                qv90 = float(cum_qv[i + 1] - cum_qv[max(a, i + 1 - RANK_BARS)])
                rows.append({"month": mp, "close": float(close[i]),
                             "bar_date": df.index[i], "n_bars": n_bars,
                             "qv90": qv90})
            recs.append({"symbol": sym, "sleeve": k, "name": "%s#%d" % (sym, k),
                         "first_bar": df.index[a], "last_bar": df.index[b - 1],
                         "n_bars_total": b - a, "rows": rows,
                         "months": months})
    return recs


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
def build_panel(recs, first_earn, last_earn):
    """Month-end closes, returns, eligibility, point-in-time membership, weights.

    Index convention: a state and a membership decided at month-end m-1 earn in
    return month m. Everything returned is aligned on RETURN months.
    """
    months = sorted(set().union(*[set(r["months"]) for r in recs]))
    months = pd.PeriodIndex(months, freq="M")
    mpos = {m: i for i, m in enumerate(months)}
    M, NS = len(months), len(recs)
    names = [r["name"] for r in recs]

    C = np.full((M, NS), np.nan)        # month-end close
    NB = np.zeros((M, NS), dtype=int)   # bars in the sleeve ending at that month-end
    QV = np.full((M, NS), np.nan)       # 90-bar quote volume at that month-end
    IN = np.zeros((M, NS), bool)        # the month is inside the sleeve
    NME = np.zeros((M, NS), dtype=int)  # month-end closes so far, inclusive
    for s, r in enumerate(recs):
        for j, row in enumerate(r["rows"]):
            i = mpos[row["month"]]
            C[i, s] = row["close"]
            NB[i, s] = row["n_bars"]
            QV[i, s] = row["qv90"]
            IN[i, s] = True
            NME[i, s] = j + 1

    # one contiguous run of in-sleeve months per sleeve, by construction
    for s in range(NS):
        v = np.flatnonzero(IN[:, s])
        if len(v) and (v[-1] - v[0] + 1) != len(v):
            raise AssertionError("sleeve %s is not contiguous in month space" % names[s])

    # eligibility and the rule state, as of signal month-end sig, aligned to m
    eligible = np.zeros((M, NS), bool)     # eligible AT month-end m (signal frame)
    state_sig = np.zeros((M, NS), bool)    # rule state decided AT month-end m
    for m in range(M):
        have10 = IN[m] & (NME[m] >= MIN_MONTH_ENDS) & (NB[m] >= MIN_BARS)
        eligible[m] = have10
        if m >= LOOKBACK - 1:
            win = C[m - LOOKBACK + 1: m + 1, :]
            full = np.isfinite(win).all(axis=0)
            ma = np.where(full, win.mean(axis=0), np.nan)
            state_sig[m] = have10 & full & (C[m] > ma)   # strictly above

    # sleeve return in month m, inside the sleeve only
    A_s = np.full((M, NS), np.nan)
    both = IN[1:] & IN[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        A_s[1:][both] = (C[1:][both] / C[:-1][both]) - 1.0

    # point-in-time membership: top TOP_N eligible at sig = m-1, ties by name
    member = np.zeros((M, NS), bool)
    w = np.zeros((M, NS))
    forced_exit = np.zeros((M, NS), bool)
    for m in range(1, M):
        sig = m - 1
        cand = np.flatnonzero(eligible[sig])
        if len(cand) < 2:
            continue
        order = sorted(cand, key=lambda s: (-QV[sig, s], names[s]))
        sel = order[:TOP_N] if len(order) > TOP_N else order
        member[m, sel] = True
        w[m, sel] = 1.0 / len(sel)
        for s in sel:
            if not IN[m, s]:
                # the sleeve ended at sig: sold at its last close, which IS the
                # sig month-end close, so the month's return is exactly zero
                if NME[sig, s] != NME[:, s].max():
                    raise AssertionError(
                        "member %s has no close in %s yet the sleeve continues: "
                        "a return would cross a gap — HALT" % (names[s], months[m]))
                A_s[m, s] = 0.0
                forced_exit[m, s] = True

    scorable = member.any(axis=1) & (member.sum(axis=1) >= 2)
    in_win = np.array([(mp >= first_earn) and (mp <= last_earn) for mp in months])
    scorable = scorable & in_win
    if not scorable.any():
        raise AssertionError("no scorable month inside the registered window")
    first = int(np.flatnonzero(scorable)[0])
    if first < 1:
        raise AssertionError("the first scored month has no signal month before it")
    last = int(np.flatnonzero(scorable)[-1])
    if not scorable[first:last + 1].all():
        raise AssertionError("the scored window is not contiguous")
    sl = slice(first, last + 1)

    assert_no_cross_gap_return(C, A_s, IN, member, forced_exit, names, months)

    tot = w[sl].sum(axis=1)
    if not np.allclose(tot, 1.0):
        raise AssertionError("target weights do not sum to 1 (max dev %.3e)"
                             % np.abs(tot - 1).max())
    bad = member[sl] & ~np.isfinite(A_s[sl])
    if bad.any():
        raise AssertionError("%d member sleeve-months carry a non-finite return"
                             % int(bad.sum()))

    # assets: one per symbol that is ever a member, plus cash
    ever = np.flatnonzero(member[sl].any(axis=0))
    syms = list(dict.fromkeys(recs[s]["symbol"] for s in ever))
    assets = syms + [CASH]
    aidx = {a: i for i, a in enumerate(assets)}
    cash_j = aidx[CASH]
    NJ = len(assets)

    keep = ever
    if not np.allclose(w[sl][:, [s for s in range(NS) if s not in set(keep.tolist())]], 0.0):
        raise AssertionError("a pruned sleeve carries a non-zero weight")
    asset_of = np.array([aidx[recs[s]["symbol"]] for s in keep])
    P = np.zeros((len(keep), NJ))
    P[np.arange(len(keep)), asset_of] = 1.0

    A = np.zeros((last - first + 1, NJ))
    for k, s in enumerate(keep):
        col = A_s[sl, s]
        j = asset_of[k]
        f = np.isfinite(col)
        A[f, j] = col[f]

    first_elig_sig = int(np.flatnonzero(eligible.sum(axis=1) >= 2)[0]) \
        if (eligible.sum(axis=1) >= 2).any() else None

    return {
        "sleeve_names": [names[s] for s in keep],
        "all_sleeve_names": names,
        "assets": assets, "cash_j": cash_j, "P": P, "asset_of": asset_of,
        "months": months[sl], "all_months": months,
        "first_idx": first, "last_idx": last,
        "w": w[sl][:, keep], "member": member[sl][:, keep],
        "state_rule": state_sig[np.arange(first - 1, last)][:, keep],
        "A": A, "A_sleeve": A_s[sl][:, keep],
        "forced_exit": forced_exit[sl][:, keep],
        "n_members": member[sl].sum(axis=1),
        "eligible_count_signal": eligible.sum(axis=1),
        "first_two_eligible_month_end": (str(months[first_elig_sig])
                                         if first_elig_sig is not None else None),
        "first_scored_month": str(months[first]),
        "last_scored_month": str(months[last]),
        "sleeves_in_panel": int(len(keep)),
        "sleeves_total": NS,
        "forced_exit_member_months": int(forced_exit[sl][:, keep].sum()),
    }


def assert_no_cross_gap_return(C, A_s, IN, member, forced_exit, names, months):
    """A finite sleeve return must have both of its closes inside one sleeve.

    This is the registered HALT 'return_computed_across_a_gap'. The only finite
    return permitted without a prior-month close is a registered forced exit,
    which is exactly zero.
    """
    f = np.isfinite(A_s)
    f[0, :] = False
    ok = np.zeros_like(f)
    ok[1:] = IN[1:] & IN[:-1]
    offend = f & ~ok & ~forced_exit
    if offend.any():
        m, s = np.argwhere(offend)[0]
        raise AssertionError(
            "return computed across a gap: sleeve %s month %s — HALT"
            % (names[s], months[m]))
    fe = forced_exit & (A_s != 0.0)
    if fe.any():
        m, s = np.argwhere(fe)[0]
        raise AssertionError(
            "forced exit for %s in %s must be a zero return, got %r"
            % (names[s], months[m], A_s[m, s]))
    return True


# ---------------------------------------------------------------------------
# Book evaluation
# ---------------------------------------------------------------------------
def evaluate_books(panel, states, bps):
    """Net monthly returns for a stack of books. states: (R, M, NS) bool.

    Weights drift with realised returns inside the month; turnover is measured
    between the drifted weights and the next month's target and charged to the
    month the new positions earn. Turnover is aggregated BY ASSET: every sleeve
    sitting in USDT is one USDT position. The first scored month's drifted
    weights are zero, so both books pay a full 1.0 buy-in.
    """
    w, P, A, cash_j = panel["w"], panel["P"], panel["A"], panel["cash_j"]
    S = np.asarray(states, dtype=float)
    if S.ndim == 2:
        S = S[None, :, :]
    WS = (w[None, :, :] * S) @ P
    WS[:, :, cash_j] += (w[None, :, :] * (1.0 - S)).sum(axis=2)
    W = WS

    tot = W.sum(axis=2)
    if not np.allclose(tot, 1.0, atol=1e-9):
        raise AssertionError("book weights do not sum to 1")

    gross = (W * A[None, :, :]).sum(axis=2)
    V = W * (1.0 + A[None, :, :])
    norm = V.sum(axis=2, keepdims=True)
    D = np.zeros_like(W)
    D[:, 1:, :] = V[:, :-1, :] / norm[:, :-1, :]
    turnover = np.abs(W - D).sum(axis=2)
    net = gross - (bps * 1e-4) * turnover
    return net, gross, turnover


def annualised_sharpe(r, ddof=1):
    """mean/std * sqrt(12). Cash is USDT at zero yield, so r IS excess return."""
    e = np.asarray(r, dtype=float)
    mu = e.mean(axis=-1)
    sd = e.std(axis=-1, ddof=ddof)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mu / sd * np.sqrt(MONTHS_PER_YEAR), np.nan)


def dsr(net_trend, net_bh):
    return annualised_sharpe(net_trend) - annualised_sharpe(net_bh)


def dmu(net_trend, net_bh):
    """Annualised arithmetic mean monthly return difference. The materiality bar."""
    return (np.asarray(net_trend, dtype=float).mean(axis=-1)
            - np.asarray(net_bh, dtype=float).mean(axis=-1)) * MONTHS_PER_YEAR


def book_metrics(net):
    net = np.asarray(net, dtype=float)
    cum = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(cum)
    n = len(net)
    return {
        "cagr": float(cum[-1] ** (MONTHS_PER_YEAR / n) - 1.0),
        "ann_vol": float(net.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)),
        "sharpe": float(annualised_sharpe(net)),
        "max_drawdown": float((cum / peak - 1.0).min()),
        "mean_monthly": float(net.mean()),
        "months": int(n),
    }


# ---------------------------------------------------------------------------
# The timing null
# ---------------------------------------------------------------------------
def measure_markov_rates(states, member):
    """Per-sleeve two-state rates from a book's own path over MEMBER month pairs.

    Frequency, never timing: this reveals how often the book switches, not when.
    """
    S = np.asarray(states, bool)
    NS = S.shape[1]
    p01 = np.zeros(NS)
    p10 = np.zeros(NS)
    pi1 = np.zeros(NS)
    pair = member[1:] & member[:-1]
    prev, cur = S[:-1], S[1:]
    for s in range(NS):
        m = pair[:, s]
        pv, cv = prev[m, s], cur[m, s]
        n0 = int((~pv).sum())
        n1 = int(pv.sum())
        p01[s] = (cv[~pv].mean() if n0 else 1.0)
        p10[s] = ((~cv[pv]).mean() if n1 else 0.0)
        tot = p01[s] + p10[s]
        if tot > 0:
            pi1[s] = p01[s] / tot
        else:
            vm = member[:, s]
            pi1[s] = float(S[vm, s].mean()) if vm.any() else 0.0
    return {"p01": p01, "p10": p10, "pi1": pi1}


def simulate_markov(rates, member, R, rng):
    """(R, M, NS) states from the per-sleeve chain; False where not a member."""
    M, NS = member.shape
    p01, p10, pi1 = rates["p01"], rates["p10"], rates["pi1"]
    first = np.full(NS, -1)
    for s in range(NS):
        v = np.flatnonzero(member[:, s])
        if len(v):
            first[s] = v[0]
    out = np.zeros((R, M, NS), bool)
    prev = np.zeros((R, NS), bool)
    for m in range(M):
        u = rng.random((R, NS))
        nxt = np.where(prev, u < (1.0 - p10)[None, :], u < p01[None, :])
        cur = np.where((first == m)[None, :], u < pi1[None, :], nxt)
        cur = cur & member[m][None, :]
        out[:, m, :] = cur
        prev = cur
    return out


def null_sharpes(panel, states_obs, bps, R, rng):
    """Trend-side Sharpes: the book under test, and R random books at its rates.

    dSR subtracts the SAME buy-and-hold Sharpe from both sides, so differences,
    the null's spread and every p are identical whether computed on Sharpes or on
    dSR — and this routine never forms the buy-and-hold book.
    """
    member = panel["member"]
    obs = states_obs if np.asarray(states_obs).ndim == 3 else np.asarray(states_obs)[None]
    net_obs, _, _ = evaluate_books(panel, obs, bps)
    s_obs = float(annualised_sharpe(net_obs[0]))
    rates = measure_markov_rates(np.asarray(obs[0], bool), member)
    S = simulate_markov(rates, member, R, rng)
    net_null, _, _ = evaluate_books(panel, S, bps)
    s_null = annualised_sharpe(net_null)
    good = np.isfinite(s_null)
    return {"sharpe_obs": s_obs, "sharpe_null": s_null[good], "rates": rates,
            "R_effective": int(good.sum())}


def timing_p(panel, states_obs, bps, R, rng):
    """p = share of random-timing books whose dSR reaches the book under test."""
    o = null_sharpes(panel, states_obs, bps, R, rng)
    p = float((o["sharpe_null"] >= o["sharpe_obs"]).mean())
    return {"p": p, "sharpe_obs": o["sharpe_obs"], "sharpe_null": o["sharpe_null"],
            "rates": o["rates"], "R_effective": o["R_effective"]}


def band_ok(share, band):
    """The known-null control passes only INSIDE the closed band [lo, hi]."""
    lo, hi = float(band[0]), float(band[1])
    return bool(lo <= float(share) <= hi)


def detection_ok(detection, floor):
    """The positive control passes at or above its floor."""
    return bool(float(detection) >= float(floor))


def mde_from_null(null_values, z=Z80):
    """(p95 - median) + z * sd. Invariant to any constant offset, so the null's
    trend-side Sharpes give the same answer as its dSRs."""
    d = np.asarray(null_values, dtype=float)
    d = d[np.isfinite(d)]
    return float((np.percentile(d, 95) - np.median(d)) + z * d.std(ddof=1))


# ---------------------------------------------------------------------------
# Planted positive control — the leak is deliberate
# ---------------------------------------------------------------------------
def plant_states(panel, q, R, rng):
    """States set to the sign of the sleeve's NEXT-MONTH return with prob q.

    DELIBERATE LEAK. The planted state reads the sign of the very return it is
    about to earn. That is the point: it proves the instrument can see a timing
    edge of the registered size. It is never a result.
    """
    member, A_s = panel["member"], panel["A_sleeve"]
    correct = A_s > 0
    keep = rng.random((R,) + member.shape) < q
    st = np.where(keep, correct[None, :, :], ~correct[None, :, :])
    return st & member[None, :, :]


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------
def invested_share(panel, states):
    """Share of member sleeve-months in the coin, and the weighted invested share."""
    member, w = panel["member"], panel["w"]
    S = np.asarray(states, bool)
    return {
        "sleeve_month_share": float(S[member].mean()),
        "weighted_share": float((w * S).sum(axis=1).mean()),
    }


def switch_count(panel, states):
    member = panel["member"]
    S = np.asarray(states, bool)
    pair = member[1:] & member[:-1]
    return int((S[1:] != S[:-1])[pair].sum())
