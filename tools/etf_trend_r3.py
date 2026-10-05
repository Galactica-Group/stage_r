"""
================================================================================
Script:       etf_trend_r3.py
Version:      1.0.0
Phase:        Stage R — Timing Triage
Purpose:      Engine for look 3 of Q4 2026: an ETF 10-month trend book held
              long-or-cash against buy-and-hold of the same ETFs, scored by the
              difference in annualised Sharpe and voted on by a random-timing
              null matched to the book's own switching frequency.
Entry Point:  No (imported by tools/run_t_r3.py)
Dependencies: numpy, pandas, pyarrow
Last Sprint:  Sprint R3 — Look 3 of Q4 (2026-10-05)
================================================================================

Every function here is pure: it takes arrays and returns arrays. Nothing reads
the network, and the only disk read is load_fund_prices, which filters the half-A
window at read time and asserts the window after the load.

The null is a TIMING null, not a no-information null. Each sleeve switches as a
two-state Markov chain at that sleeve's own rates measured under the rule, so a
random book carries the rule's cash level and the rule's turnover and differs
only in WHEN it is out. Measuring the rates reveals frequency, not performance.
A no-information null against buy-and-hold would sit below zero, because random
timing at invested share k carries a Sharpe near sqrt(k) of buy-and-hold; beating
that null would be unearned.
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = "1.0.0"

CLASSES = {
    "us_equity": ["SPY", "IWM", "QQQ"],
    "intl_equity": ["EFA", "EEM"],
    "bonds": ["TLT", "IEF", "TIP", "LQD", "HYG"],
    "real_estate": ["IYR", "VNQ"],
    "commodities": ["GLD", "SLV", "DBC", "USO"],
}
CASH = "SHY"
LOOKBACK = 10
BPS_BASE = 5.0
BPS_STRESS = 15.0
MONTHS_PER_YEAR = 12
Z80 = 0.8416212335729143  # one-sided z at 80% power


# ---------------------------------------------------------------------------
# Loading — the only disk-touching function, and the only place the window lives
# ---------------------------------------------------------------------------
def load_fund_prices(vintage_dir, tickers, exclusive_max):
    """Daily closeadj per ticker, filtered to dates strictly before exclusive_max.

    The filter is applied at read time AND asserted after the load. The raw span
    is also asserted to reach past exclusive_max, so a short slice is a vendor
    truncation rather than the window doing its job.
    """
    from pathlib import Path
    cut = pd.Timestamp(exclusive_max)
    out, spans = {}, {}
    for t in tickers:
        p = Path(vintage_dir) / ("etf_%s.parquet" % t)
        df = pd.read_parquet(p)
        d = pd.to_datetime(df["date"])
        raw_max = d.max()
        if not (raw_max >= cut):
            raise AssertionError(
                "%s raw span ends %s, before the window edge %s: the half-A slice "
                "would be a vendor truncation, not a filter" % (t, raw_max.date(), cut.date()))
        keep = d < cut
        s = pd.Series(df.loc[keep, "closeadj"].astype(float).to_numpy(),
                      index=d[keep].to_numpy()).sort_index()
        if len(s) and not (s.index.max() < cut):
            raise AssertionError("%s loaded a row at or past %s" % (t, cut.date()))
        out[t] = s
        spans[t] = (str(s.index.min().date()), str(s.index.max().date()), int(len(s)),
                    str(raw_max.date()))
    return out, spans


def monthly_last(daily):
    """Last observation of each calendar month, indexed by month period."""
    g = daily.groupby(pd.PeriodIndex(daily.index, freq="M"))
    return g.last()


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
def build_panel(prices, classes=None, cash=CASH, lookback=LOOKBACK):
    """Assemble the month-end close matrix, returns, states, eligibility, weights.

    Index convention: a state decided at month-end m-1 earns in return month m.
    Everything returned is aligned on RETURN months.
    """
    classes = CLASSES if classes is None else classes
    sleeves = [t for c in classes for t in classes[c]]
    cls_of = np.array([i for i, c in enumerate(classes) for _ in classes[c]])
    n_cls = len(classes)

    assets = list(dict.fromkeys(sleeves + [cash]))
    aidx = {a: i for i, a in enumerate(assets)}
    cash_j = aidx[cash]
    asset_of = np.array([aidx[t] for t in sleeves])

    monthly = {t: monthly_last(prices[t]) for t in assets}
    months = sorted(set().union(*[set(m.index) for m in monthly.values()]))
    months = pd.PeriodIndex(months, freq="M")
    M, NS, NJ = len(months), len(sleeves), len(assets)

    C = np.full((M, NS), np.nan)
    for s, t in enumerate(sleeves):
        C[:, s] = monthly[t].reindex(months).to_numpy(dtype=float)
    CA = np.full((M, NJ), np.nan)
    for a, t in enumerate(assets):
        CA[:, a] = monthly[t].reindex(months).to_numpy(dtype=float)

    # asset returns on return-month m
    A = np.full((M, NJ), np.nan)
    A[1:] = CA[1:] / CA[:-1] - 1.0

    # rule state and eligibility, as of signal month m-1, aligned to return month m
    valid = np.zeros((M, NS), bool)
    state_rule = np.zeros((M, NS), bool)
    for m in range(M):
        sig = m - 1
        if sig < lookback - 1:
            continue
        win = C[sig - lookback + 1: sig + 1, :]
        have = np.isfinite(win).all(axis=0) & np.isfinite(C[m, :])
        ma = np.where(have, win.mean(axis=0), np.nan)
        valid[m] = have
        state_rule[m] = have & (C[sig] > ma)

    shy_r = A[:, cash_j]

    # classes present and the scored window
    cls_present = np.zeros((M, n_cls), bool)
    for k in range(n_cls):
        cls_present[:, k] = valid[:, cls_of == k].any(axis=1)
    n_present = cls_present.sum(axis=1)
    scorable = (n_present >= 2) & np.isfinite(shy_r)

    if not scorable.any():
        raise AssertionError("no scorable month")
    first = int(np.flatnonzero(scorable)[0])
    last = int(np.flatnonzero(scorable)[-1])
    if not scorable[first:last + 1].all():
        raise AssertionError("scored window is not contiguous")
    sl = slice(first, last + 1)

    # target weights on the scored window
    w = np.zeros((M, NS))
    for m in range(first, last + 1):
        cw = 1.0 / n_present[m]
        for k in range(n_cls):
            mem = np.flatnonzero((cls_of == k) & valid[m])
            if len(mem):
                w[m, mem] = cw / len(mem)
    tot = w[sl].sum(axis=1)
    if not np.allclose(tot, 1.0):
        raise AssertionError("target weights do not sum to 1 (max dev %.3e)"
                             % np.abs(tot - 1).max())

    # an eligible sleeve must have a finite return, else a weight meets a NaN
    bad = valid[sl] & ~np.isfinite(A[sl][:, asset_of])
    if bad.any():
        raise AssertionError("%d eligible sleeve-months carry a non-finite return"
                             % int(bad.sum()))

    P = np.zeros((NS, NJ))
    P[np.arange(NS), asset_of] = 1.0

    A_safe = np.nan_to_num(A[sl], nan=0.0)

    return {
        "classes": list(classes), "sleeves": sleeves, "assets": assets,
        "cls_of": cls_of, "asset_of": asset_of, "cash_j": cash_j, "P": P,
        "months": months[sl], "all_months": months,
        "first_idx": first, "last_idx": last,
        "w": w[sl], "valid": valid[sl], "state_rule": state_rule[sl],
        "A": A_safe, "shy_r": shy_r[sl],
        "n_present": n_present[sl], "cls_present": cls_present,
        "entry_month": {c: (str(months[np.flatnonzero(cls_present[:, k])[0]])
                            if cls_present[:, k].any() else None)
                        for k, c in enumerate(classes)},
        "first_two_class_month": str(months[int(np.flatnonzero(n_present >= 2)[0])])
                                 if (n_present >= 2).any() else None,
        "first_cash_return_month": str(months[int(np.flatnonzero(np.isfinite(shy_r))[0])])
                                   if np.isfinite(shy_r).any() else None,
    }


# ---------------------------------------------------------------------------
# Book evaluation
# ---------------------------------------------------------------------------
def evaluate_books(panel, states, bps):
    """Net monthly returns for a stack of books. states: (R, M, NS) bool.

    Weights drift with realised returns inside the month; turnover is measured
    between the drifted weights and the next month's target and charged to the
    month the new positions earn. Turnover is aggregated BY ASSET: two sleeves
    sitting in cash are one cash position.
    """
    w, P, A, cash_j = panel["w"], panel["P"], panel["A"], panel["cash_j"]
    S = states.astype(float)
    WS = w[None, :, :] * S
    W = WS @ P
    W[:, :, cash_j] += (w[None, :, :] * (1.0 - S)).sum(axis=2)

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


def annualised_sharpe(excess, ddof=1):
    """mean/std * sqrt(12) on monthly excess returns. Vectorised on the last axis."""
    e = np.asarray(excess, dtype=float)
    mu = e.mean(axis=-1)
    sd = e.std(axis=-1, ddof=ddof)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mu / sd * np.sqrt(MONTHS_PER_YEAR), np.nan)
    return out


def dsr(net_trend, net_bh, shy_r):
    """TREND annualised Sharpe minus BUY-AND-HOLD annualised Sharpe, excess of cash."""
    return (annualised_sharpe(net_trend - shy_r)
            - annualised_sharpe(net_bh - shy_r))


def book_metrics(net, shy_r):
    net = np.asarray(net, dtype=float)
    cum = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(cum)
    mdd = float((cum / peak - 1.0).min())
    n = len(net)
    cagr = float(cum[-1] ** (MONTHS_PER_YEAR / n) - 1.0)
    w12 = np.nan
    if n >= 12:
        roll = np.array([np.prod(1.0 + net[i:i + 12]) - 1.0 for i in range(n - 11)])
        w12 = float(roll.min())
    return {
        "cagr": cagr,
        "ann_vol": float(net.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)),
        "sharpe_excess_cash": float(annualised_sharpe(net - shy_r)),
        "max_drawdown": mdd,
        "worst_rolling_12m": w12,
        "months": int(n),
    }


# ---------------------------------------------------------------------------
# The timing null
# ---------------------------------------------------------------------------
def measure_markov_rates(states, valid):
    """Per-sleeve two-state transition rates from a book's own state path.

    Counted over consecutive month pairs in which the sleeve is eligible on both.
    This reveals how OFTEN the book switches, never WHEN.
    """
    S = states.astype(bool)
    NS = S.shape[1]
    p01 = np.zeros(NS)
    p10 = np.zeros(NS)
    pi1 = np.zeros(NS)
    pair = valid[1:] & valid[:-1]
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
            vm = valid[:, s]
            pi1[s] = float(S[vm, s].mean()) if vm.any() else 0.0
    return {"p01": p01, "p10": p10, "pi1": pi1}


def simulate_markov(rates, valid, R, rng):
    """(R, M, NS) states drawn from the per-sleeve chain; zero where ineligible."""
    M, NS = valid.shape
    p01, p10, pi1 = rates["p01"], rates["p10"], rates["pi1"]
    first = np.full(NS, -1)
    for s in range(NS):
        v = np.flatnonzero(valid[:, s])
        if len(v):
            first[s] = v[0]
    out = np.zeros((R, M, NS), bool)
    prev = np.zeros((R, NS), bool)
    for m in range(M):
        u = rng.random((R, NS))
        nxt = np.where(prev, u < (1.0 - p10)[None, :], u < p01[None, :])
        cur = np.where((first == m)[None, :], u < pi1[None, :], nxt)
        cur = cur & valid[m][None, :]
        out[:, m, :] = cur
        prev = cur
    return out


def timing_p(panel, states_obs, bps, R, rng, net_bh=None):
    """p = share of random-timing books whose dSR reaches the book under test.

    The rates are measured from the book UNDER TEST, so the null carries that
    book's own cash level and turnover. The same procedure serves the observed
    book, every known-null control draw and every planted draw.
    """
    valid, shy = panel["valid"], panel["shy_r"]
    if net_bh is None:
        net_bh, _, _ = evaluate_books(panel, valid[None, :, :], bps)
        net_bh = net_bh[0]
    obs = states_obs if states_obs.ndim == 3 else states_obs[None, :, :]
    net_obs, _, _ = evaluate_books(panel, obs, bps)
    d_obs = float(dsr(net_obs[0], net_bh, shy))

    rates = measure_markov_rates(obs[0], valid)
    S = simulate_markov(rates, valid, R, rng)
    net_null, _, _ = evaluate_books(panel, S, bps)
    d_null = dsr(net_null, net_bh[None, :], shy[None, :])
    good = np.isfinite(d_null)
    p = float((d_null[good] >= d_obs).mean())
    return {"dsr": d_obs, "p": p, "null": d_null[good], "rates": rates,
            "R_effective": int(good.sum())}


def mde_from_null(d_null, z=Z80):
    """The dSR detectable at 80% power against this null's own spread."""
    d = np.asarray(d_null, dtype=float)
    d = d[np.isfinite(d)]
    return float((np.percentile(d, 95) - np.median(d)) + z * d.std(ddof=1))


# ---------------------------------------------------------------------------
# Planted positive control — the leak is deliberate
# ---------------------------------------------------------------------------
def plant_states(panel, q, R, rng):
    """States set to the sign of the NEXT-MONTH excess return with probability q.

    DELIBERATE LEAK. The state reads the sign of the very return it is about to
    earn, which is exactly why it must be detected. It exists to prove the
    instrument can see a real timing edge, and must never be read as a result.
    """
    valid, A, shy = panel["valid"], panel["A"], panel["shy_r"]
    ex = A[:, panel["asset_of"]] - shy[:, None]
    correct = ex > 0
    keep = rng.random((R,) + valid.shape) < q
    st = np.where(keep, correct[None, :, :], ~correct[None, :, :])
    return st & valid[None, :, :]


# ---------------------------------------------------------------------------
# Reported-not-voting interval
# ---------------------------------------------------------------------------
def stationary_block_indices(M, mean_block, R, rng):
    """Stationary (Politis-Romano) block bootstrap index matrix, wrapped."""
    p = 1.0 / mean_block
    idx = np.empty((R, M), dtype=np.int64)
    cur = rng.integers(0, M, size=R)
    idx[:, 0] = cur
    for t in range(1, M):
        fresh = rng.random(R) < p
        cur = np.where(fresh, rng.integers(0, M, size=R), (cur + 1) % M)
        idx[:, t] = cur
    return idx


def paired_bootstrap(net_trend, net_bh, shy_r, mean_block, R, rng):
    """One shared month index applied to both books: the pairing is the point."""
    M = len(net_trend)
    idx = stationary_block_indices(M, mean_block, R, rng)
    et, eb = net_trend - shy_r, net_bh - shy_r
    d = annualised_sharpe(et[idx]) - annualised_sharpe(eb[idx])
    dm = (net_trend - net_bh)[idx].mean(axis=1)
    f = np.isfinite(d)
    return {
        "dsr_ci90": [float(np.percentile(d[f], 5)), float(np.percentile(d[f], 95))],
        "mean_monthly_diff_ci90": [float(np.percentile(dm, 5)), float(np.percentile(dm, 95))],
        "mean_monthly_diff_point": float((net_trend - net_bh).mean()),
        "R": int(f.sum()), "mean_block": mean_block,
    }


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------
def cash_time_by_class(panel, states):
    """Share of eligible sleeve-months spent in cash, per class."""
    valid, cls_of, classes = panel["valid"], panel["cls_of"], panel["classes"]
    S = np.asarray(states, bool)
    out = {}
    for k, c in enumerate(classes):
        m = valid & (cls_of == k)[None, :]
        out[c] = float((~S[m]).mean()) if m.any() else None
    allm = valid
    out["_all"] = float((~S[allm]).mean())
    return out


def switch_count(panel, states):
    valid = panel["valid"]
    S = np.asarray(states, bool)
    pair = valid[1:] & valid[:-1]
    return int((S[1:] != S[:-1])[pair].sum())
