"""
================================================================================
Script:       xs_triage.py
Version:      1.0.0
Phase:        Stage R — Cross-Sectional Triage
Purpose:      Reusable cross-sectional triage harness: point-in-time universe
              construction, quintile portfolio arithmetic, provisional costs,
              and the voting / reported nulls with their controls.
Entry Point:  No (imported by tools/build_panel_r1.py and tools/run_t_r1.py)
Dependencies: numpy, pandas (no factory imports, no scipy)
Last Sprint:  Sprint R1 — Cross-Sectional Triage Harness (2026-10-04)
================================================================================

The harness separates three concerns: (1) window guards that refuse any daily
row at or beyond the half-A cutoff, (2) deterministic portfolio and cost
arithmetic over a monthly panel, (3) inference — a stationary block bootstrap
of months on the demeaned long-short net series as the VOTING null, and a
within-date rank permutation REPORTED beside it for continuity only.
No Stage R statistic is computed from a vendor price that leaves this process.
================================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

VERSION = "1.0.0"

# ------------------------------------------------------------------ constants
# Normal quantiles, hardcoded (scipy is not a Stage R dependency).
Z_95 = 1.6448536269514722   # one-sided alpha = 0.05
Z_80 = 0.8416212335729143   # power = 0.80
MONTHS_PER_YEAR = 12

# The five registered signals and their pre-registered long side.
# "HIGH" => long the top of the cross-section; "LOW" => long the bottom.
REGISTERED_SIGNALS: dict[str, str] = {
    "MOM12_1": "HIGH",
    "REV1": "LOW",
    "LOWVOL": "LOW",
    "HI52": "HIGH",
    "MAX": "LOW",
}

# Universe rules, in the registered order of application.
MIN_PRICE = 5.00
LOOKBACK_SESSIONS = 21
MIN_SESSIONS_TRADED = 15
TOP_N_BY_DOLLAR_VOLUME = 1000
VOL_WINDOW_SESSIONS = 252
VOL_MIN_PRESENT = 200

N_QUINTILES = 5


# -------------------------------------------------------------- window guards
class WindowViolation(AssertionError):
    """A daily row at or beyond the half-A cutoff reached a load."""


def assert_no_rows_at_or_after(dates, exclusive_max: str, where: str) -> pd.Timestamp | None:
    """Raise WindowViolation if any date is >= exclusive_max. Returns the max date seen."""
    cutoff = pd.Timestamp(exclusive_max)
    s = pd.to_datetime(pd.Series(np.asarray(dates)).dropna())
    if len(s) == 0:
        return None
    mx = s.max()
    if mx >= cutoff:
        bad = int((s >= cutoff).sum())
        raise WindowViolation(
            f"{where}: {bad} row(s) dated on or after {exclusive_max}; max seen {mx.date()}"
        )
    return mx


def filter_to_half_a(df: pd.DataFrame, date_col: str, exclusive_max: str, where: str) -> pd.DataFrame:
    """Filter at read time, then assert. Both halves of the guard, always together."""
    cutoff = pd.Timestamp(exclusive_max)
    out = df.loc[pd.to_datetime(df[date_col]) < cutoff].copy()
    assert_no_rows_at_or_after(out[date_col], exclusive_max, where)
    return out


# ------------------------------------------------------- permaticker identity
def sep_ticker_map(tickers: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict the securities master to table == 'SEP' and return the
    ticker -> permaticker map, asserting it is 1:1 within the vintage.
    """
    sep = tickers.loc[tickers["table"].astype("string") == "SEP"].copy()
    if sep.empty:
        raise ValueError("no SEP rows in the securities master")
    n_tick = sep["ticker"].nunique()
    n_perma = sep["permaticker"].nunique()
    if len(sep) != n_tick:
        raise ValueError(f"ticker not unique within SEP: {len(sep)} rows, {n_tick} tickers")
    if len(sep) != n_perma:
        raise ValueError(f"permaticker not unique within SEP: {len(sep)} rows, {n_perma} permatickers")
    return sep


def admitted_common_stock(sep: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rule 1: common stock only — category containing 'Common Stock'."""
    cat = sep["category"].astype("string").fillna("")
    keep = cat.str.contains("Common Stock", case=True, regex=False)
    counts = cat.loc[keep].value_counts().to_dict()
    return sep.loc[keep].copy(), {str(k): int(v) for k, v in counts.items()}


def apply_universe_rules(df: pd.DataFrame, price_col: str = "closeunadj_t",
                         sessions_col: str = "n_sessions_21",
                         dollar_vol_col: str = "med_dollar_vol_21",
                         month_col: str = "month",
                         top_n: int = TOP_N_BY_DOLLAR_VOLUME) -> tuple[pd.Series, pd.Series]:
    """
    Universe rules 2, 3 and 4, applied in the registered order. Rule 1 (common
    stock only) is applied upstream, at the securities master.

      2. closeunadj >= 5.00 at t
      3. trading on at least 15 of the last 21 GLOBAL sessions
      4. the top `top_n` by median dollar volume over the last 21 sessions

    Returns (candidate, in_universe). Rule 4 is cross-sectional and is resolved
    inside each month among the names that cleared 2 and 3.
    """
    price_ok = df[price_col].to_numpy() >= MIN_PRICE
    sessions_ok = df[sessions_col].to_numpy() >= MIN_SESSIONS_TRADED
    dv_ok = np.isfinite(df[dollar_vol_col].to_numpy())
    cand = pd.Series(price_ok & sessions_ok & dv_ok, index=df.index)

    in_universe = pd.Series(False, index=df.index)
    sub = df.loc[cand]
    if not sub.empty:
        rank = sub.groupby(month_col, observed=True)[dollar_vol_col].rank(
            ascending=False, method="first")
        in_universe.loc[sub.index[(rank <= top_n).to_numpy()]] = True
    return cand, in_universe


# --------------------------------------------------------- portfolio plumbing
def quintiles_by_month(values: np.ndarray) -> np.ndarray:
    """
    Rank into 1..N_QUINTILES by ascending value (1 = lowest). NaN -> 0 (excluded).
    Ties are broken by position, which is deterministic for a sorted input and
    immaterial at a thousand names.
    """
    out = np.zeros(values.shape[0], dtype=np.int8)
    ok = np.isfinite(values)
    n = int(ok.sum())
    if n < N_QUINTILES:
        return out
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(values[idx], kind="stable")]
    # Equal-count buckets; the remainder lands in the lower buckets.
    edges = (np.arange(1, N_QUINTILES) * n) // N_QUINTILES
    bucket = np.searchsorted(edges, np.arange(n), side="right") + 1
    out[order] = bucket.astype(np.int8)
    return out


def long_side_quintile(direction: str) -> int:
    """The registered good side: Q5 for HIGH, Q1 for LOW."""
    if direction == "HIGH":
        return N_QUINTILES
    if direction == "LOW":
        return 1
    raise ValueError(f"unknown direction {direction!r}")


def short_side_quintile(direction: str) -> int:
    return 1 if direction == "HIGH" else N_QUINTILES


def target_weights(quintile: np.ndarray, long_q: int, short_q: int, mode: str,
                   in_universe: np.ndarray) -> np.ndarray:
    """
    RETURN weights for one month — the vector the forward returns are dotted into.
      mode 'LS'        : +1 over the long quintile, -1 over the short quintile.
      mode 'LONG_ONLY' : +1 over the long quintile, -1 over the whole universe
                         (the benchmark leg, which is measured, not traded).
    """
    w = np.zeros(quintile.shape[0], dtype=np.float64)
    lmask = quintile == long_q
    nl = int(lmask.sum())
    if nl == 0:
        return w
    w[lmask] = 1.0 / nl
    if mode == "LS":
        smask = quintile == short_q
        ns = int(smask.sum())
        if ns == 0:
            return np.zeros_like(w)
        w[smask] -= 1.0 / ns
    elif mode == "LONG_ONLY":
        umask = in_universe.astype(bool)
        nu = int(umask.sum())
        if nu == 0:
            return np.zeros_like(w)
        w[umask] -= 1.0 / nu
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return w


def traded_weights(quintile: np.ndarray, long_q: int, short_q: int, mode: str,
                   in_universe: np.ndarray) -> np.ndarray:
    """
    TRADED weights for one month — the positions actually held, which is what
    turnover and borrow are charged on.

    For LS these are the return weights: both legs are real positions.
    For LONG_ONLY only the long quintile is held; the universe-equal-weight leg
    is a benchmark we do not trade, so it is charged neither turnover nor borrow.
    """
    if mode == "LS":
        return target_weights(quintile, long_q, short_q, "LS", in_universe)
    if mode == "LONG_ONLY":
        w = np.zeros(quintile.shape[0], dtype=np.float64)
        lmask = quintile == long_q
        nl = int(lmask.sum())
        if nl == 0:
            return w
        w[lmask] = 1.0 / nl
        return w
    raise ValueError(f"unknown mode {mode!r}")


# ------------------------------------------------------------------ cost model
@dataclass(frozen=True)
class CostSpec:
    """Provisional costs. Labelled provisional everywhere they are reported."""
    label: str
    bps_per_side: float
    borrow_bps_per_year: float
    commission_bps: float = 0.0

    def monthly_borrow_rate(self) -> float:
        return (self.borrow_bps_per_year / 1e4) / MONTHS_PER_YEAR


COST_BASE = CostSpec(label="BASE", bps_per_side=5.0, borrow_bps_per_year=50.0)
COST_STRESS = CostSpec(label="STRESS", bps_per_side=15.0, borrow_bps_per_year=100.0)


def traded_notional(w_now: np.ndarray, w_prev: np.ndarray) -> float:
    """
    Gross notional traded to move from w_prev to w_now, summing both sides:
    sum |w_now - w_prev|. A sell and the matching buy are each charged once.
    """
    return float(np.abs(w_now - w_prev).sum())


def month_costs(w_traded_now: np.ndarray, w_traded_prev: np.ndarray,
                spec: CostSpec) -> tuple[float, float]:
    """
    Return (cost_fraction, traded_notional) for one month, given the TRADED
    weight vectors. Turnover is charged per side on the gross notional moved;
    borrow is charged on whatever short notional is held (none in LONG_ONLY,
    whose traded vector has no negative entries).
    """
    turn = traded_notional(w_traded_now, w_traded_prev)
    borrow_notional = float(np.abs(np.minimum(w_traded_now, 0.0)).sum())
    cost = ((spec.bps_per_side + spec.commission_bps) / 1e4) * turn
    cost += spec.monthly_borrow_rate() * borrow_notional
    return cost, turn


# ------------------------------------------------------------- series builder
@dataclass
class LegSeries:
    """One signal, one mode, one cost spec: the realised monthly series."""
    months: list[pd.Timestamp] = field(default_factory=list)
    gross: list[float] = field(default_factory=list)
    net: list[float] = field(default_factory=list)
    turnover: list[float] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"month": self.months, "gross": self.gross, "net": self.net,
             "turnover": self.turnover}
        )


def build_series(panel: pd.DataFrame, signal_col: str, direction: str, mode: str,
                 spec: CostSpec, fwd_col: str = "fwd_ret",
                 universe_col: str = "in_universe",
                 month_col: str = "month", key_col: str = "permaticker") -> LegSeries:
    """
    Walk the panel month by month, form quintiles inside the universe on the
    registered direction, and realise the next month's return.
    """
    long_q = long_side_quintile(direction)
    short_q = short_side_quintile(direction)
    out = LegSeries()
    prev_w: dict[int, float] = {}

    cols = [key_col, signal_col, fwd_col, universe_col]
    sub = panel.loc[panel[universe_col].astype(bool), [month_col] + cols]
    for month, g in sub.groupby(month_col, sort=True, observed=True):
        keys = g[key_col].to_numpy()
        sig = g[signal_col].to_numpy(dtype=np.float64)
        fwd = g[fwd_col].to_numpy(dtype=np.float64)
        univ = np.ones(len(g), dtype=bool)
        # A name with no forward return cannot be held; drop before ranking so
        # the quintiles are formed over holdable names only.
        holdable = np.isfinite(fwd)
        if holdable.sum() < N_QUINTILES:
            continue
        keys, sig, fwd, univ = keys[holdable], sig[holdable], fwd[holdable], univ[holdable]

        q = quintiles_by_month(sig)
        w_ret = target_weights(q, long_q, short_q, mode, univ)
        w_trd = traded_weights(q, long_q, short_q, mode, univ)
        if not np.any(w_ret != 0.0):
            continue

        w_prev = np.array([prev_w.get(int(k), 0.0) for k in keys], dtype=np.float64)
        # Positions held last month in names absent this month are closed; that
        # is a real trade and is charged, not silently dropped.
        here = {int(x) for x in keys}
        dropped = float(sum(abs(v) for k, v in prev_w.items() if int(k) not in here))

        gross = float(np.dot(w_ret, fwd))
        cost, turn = month_costs(w_trd, w_prev, spec)
        cost += ((spec.bps_per_side + spec.commission_bps) / 1e4) * dropped
        turn += dropped

        out.months.append(month)
        out.gross.append(gross)
        out.net.append(gross - cost)
        out.turnover.append(turn)
        prev_w = {int(k): float(v) for k, v in zip(keys, w_trd) if v != 0.0}
    return out


# ------------------------------------------------------------------- inference
def annualised_sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(x.mean() / sd * np.sqrt(MONTHS_PER_YEAR))


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """
    Politis-Romano stationary bootstrap index draw of length n with
    geometric block lengths of mean `mean_block` and wrap-around.
    """
    p = 1.0 / float(mean_block)
    idx = np.empty(n, dtype=np.int64)
    cur = int(rng.integers(0, n))
    idx[0] = cur
    jump = rng.random(n) < p
    starts = rng.integers(0, n, size=n)
    for t in range(1, n):
        if jump[t]:
            cur = int(starts[t])
        else:
            cur = (cur + 1) % n
        idx[t] = cur
    return idx


def stationary_bootstrap_index_matrix(n: int, mean_block: float, R: int,
                                      seed: int) -> np.ndarray:
    """
    ONE (R, n) matrix of month indices. The batch's five long-short series share
    this single draw, so replicate r resamples the same months for every signal
    and the five p-values are drawn against a common resampling of history.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((R, n), dtype=np.int64)
    for r in range(R):
        out[r] = stationary_bootstrap_indices(n, mean_block, rng)
    return out


def p_from_index_matrix(series: np.ndarray, idx: np.ndarray) -> dict:
    """
    VOTING NULL evaluated against a pre-drawn month-index matrix. The series is
    DEMEANED before resampling, so the null is centred at zero by construction
    and its spread does not encode the observed mean.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.size != idx.shape[1]:
        raise ValueError(f"series length {x.size} != index matrix width {idx.shape[1]}")
    d = x - x.mean()
    boots = d[idx].mean(axis=1)
    obs = float(x.mean())
    R = idx.shape[0]
    return {"n": int(x.size), "p": (1.0 + float((boots >= obs).sum())) / (R + 1.0),
            "null_sd": float(boots.std(ddof=1)), "null_mean": float(boots.mean()),
            "R": int(R), "boots": boots}


def shared_batch_nulls(series_map: dict[str, np.ndarray], mean_block: float = 3.0,
                       R: int = 2000, seed: int = 0) -> tuple[dict[str, dict], np.ndarray]:
    """
    Evaluate every series in the batch against ONE shared month-index draw.
    All series must be the same length — the batch's common month window.
    """
    lengths = {k: len(v) for k, v in series_map.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"batch series lengths differ, so one month index cannot be "
                         f"shared: {lengths}")
    n = next(iter(lengths.values()))
    idx = stationary_bootstrap_index_matrix(n, mean_block, R, seed)
    return {k: p_from_index_matrix(v, idx) for k, v in series_map.items()}, idx


def block_bootstrap_p(series: np.ndarray, mean_block: float = 3.0, R: int = 2000,
                      seed: int = 0) -> dict:
    """
    VOTING NULL. Stationary block bootstrap of MONTHS on the DEMEANED series;
    the statistic is the sample mean. One-sided p for mean > 0.

    Demeaning is what makes this a null: the resampled distribution is centred
    at zero by construction, so its spread (and the MDE derived from it) does
    not encode the observed mean.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return {"n": n, "p": float("nan"), "null_sd": float("nan"),
                "null_mean": float("nan"), "R": R, "mean_block": mean_block}
    d = x - x.mean()
    rng = np.random.default_rng(seed)
    boots = np.empty(R, dtype=np.float64)
    for r in range(R):
        boots[r] = d[stationary_bootstrap_indices(n, mean_block, rng)].mean()
    obs = float(x.mean())
    p = (1.0 + float((boots >= obs).sum())) / (R + 1.0)
    return {"n": n, "p": float(p), "null_sd": float(boots.std(ddof=1)),
            "null_mean": float(boots.mean()), "R": R, "mean_block": float(mean_block),
            "boots": boots}


def mde_from_null_sd(null_sd: float, alpha_z: float = Z_95, power_z: float = Z_80) -> float:
    """Minimum detectable monthly mean at the stated power, from the null's own spread."""
    if not np.isfinite(null_sd):
        return float("nan")
    return float((alpha_z + power_z) * null_sd)


def within_date_permutation_p(panel: pd.DataFrame, signal_col: str, direction: str,
                              mode: str, spec: CostSpec, R: int = 200, seed: int = 0,
                              fwd_col: str = "fwd_ret", universe_col: str = "in_universe",
                              month_col: str = "month", key_col: str = "permaticker") -> dict:
    """
    REPORTED, NOT VOTED. Permutes the signal across names WITHIN each date.

    Two independent reasons it does not vote, the second measured here:
      (1) it treats each month's factor realisation as noise-free and so
          overstates significance;
      (2) permuting across names destroys the signal's month-to-month
          PERSISTENCE, so the permuted book rotates far harder than the real
          one and pays far more cost. On a NET statistic that alone makes any
          persistent real signal look significant whatever its edge. The
          permuted and real turnovers are returned so the size of that artifact
          is visible, and the gross p is returned beside the net p because the
          gross comparison is at least not confounded by it.
    """
    obs = build_series(panel, signal_col, direction, mode, spec,
                       fwd_col=fwd_col, universe_col=universe_col,
                       month_col=month_col, key_col=key_col)
    obs_net = float(np.mean(obs.net)) if obs.net else float("nan")
    obs_gross = float(np.mean(obs.gross)) if obs.gross else float("nan")
    obs_turn = float(np.mean(obs.turnover)) if obs.turnover else float("nan")

    rng = np.random.default_rng(seed)
    work = panel.copy()
    ge_net = ge_gross = 0
    turns: list[float] = []
    for _ in range(R):
        work["_perm"] = work.groupby(month_col, sort=False, observed=True)[
            signal_col].transform(lambda s, _r=rng: s.to_numpy()[_r.permutation(len(s))])
        s = build_series(work, "_perm", direction, mode, spec, fwd_col=fwd_col,
                         universe_col=universe_col, month_col=month_col, key_col=key_col)
        if not s.net:
            continue
        if float(np.mean(s.net)) >= obs_net:
            ge_net += 1
        if float(np.mean(s.gross)) >= obs_gross:
            ge_gross += 1
        turns.append(float(np.mean(s.turnover)))
    return {"p": (1.0 + ge_net) / (R + 1.0),
            "p_gross": (1.0 + ge_gross) / (R + 1.0),
            "R": R, "obs_mean": obs_net, "obs_mean_gross": obs_gross,
            "obs_turnover": obs_turn,
            "permuted_turnover": float(np.mean(turns)) if turns else float("nan")}


# --------------------------------------------------------- known-null control
def median_rank_autocorr(panel: pd.DataFrame, signal_cols: list[str],
                         month_col: str = "month", key_col: str = "permaticker",
                         universe_col: str = "in_universe") -> float:
    """
    Median month-to-month cross-sectional rank autocorrelation across the batch.
    This sets phi for the AR(1) noise so the control's persistence matches the
    real signals' persistence rather than being white.
    """
    sub = panel.loc[panel[universe_col].astype(bool)]
    rhos: list[float] = []
    for col in signal_cols:
        wide = sub.pivot_table(index=month_col, columns=key_col, values=col, observed=True)
        ranks = wide.rank(axis=1, pct=True)
        a = ranks.to_numpy()
        for i in range(1, a.shape[0]):
            x, y = a[i - 1], a[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 30:
                xs, ys = x[m], y[m]
                sx, sy = xs.std(), ys.std()
                if sx > 0 and sy > 0:
                    rhos.append(float(np.corrcoef(xs, ys)[0, 1]))
    if not rhos:
        return 0.0
    return float(np.median(rhos))


def ar1_noise_signal(months: np.ndarray, keys: np.ndarray, phi: float,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Per-permaticker AR(1) noise with the given monthly phi, unit unconditional
    variance, laid onto the (month, key) index order of the panel.
    """
    order = np.lexsort((months, keys))
    out = np.empty(months.shape[0], dtype=np.float64)
    s = float(np.sqrt(max(0.0, 1.0 - phi * phi)))
    prev_key = None
    x = 0.0
    eps = rng.standard_normal(months.shape[0])
    for j, pos in enumerate(order):
        k = keys[pos]
        if k != prev_key:
            x = eps[j]          # stationary start
            prev_key = k
        else:
            x = phi * x + s * eps[j]
        out[pos] = x
    return out


def control_verdict(share: float, accept: tuple[float, float]) -> bool:
    """
    The acceptance test for a known-null control: the share of synthetic signals
    at or below alpha must lie inside the band. Pure, so both the pass and the
    fail are directly testable.
    """
    if not np.isfinite(share):
        return False
    return bool(accept[0] <= share <= accept[1])


def known_null_control(panel: pd.DataFrame, n_synthetic: int, phi: float,
                       spec: CostSpec, R: int = 500, seed: int = 0,
                       mode: str = "LS", direction: str = "HIGH",
                       accept: tuple[float, float] = (0.01, 0.10),
                       alpha: float = 0.05, mean_block: float = 3.0,
                       calibration_arm: str = "gross") -> dict:
    """
    IN RUN. n_synthetic AR(1)-noise signals, each through the FULL pipeline
    (universe, quintiles, weights, costs) at the stated R.

    CALIBRATION ARM. The size of a one-sided test of "mean > 0" is defined at
    the null boundary, where the true mean is zero. The GROSS arm realises that
    boundary: a no-edge signal has a true gross mean of zero and its sample mean
    varies around it, so the share at p <= alpha estimates the test's actual
    size. The NET arm does not: a no-edge strategy that pays costs has a true
    mean well below zero, so its share is mechanically near nil whatever the
    inference does. Both shares are reported; `pass` is adjudicated on
    `calibration_arm`, and the net share is reported as the conservative figure.
    """
    work = panel.loc[panel["in_universe"].astype(bool)].copy()
    months = work["month"].to_numpy()
    keys = work["permaticker"].to_numpy()
    rng = np.random.default_rng(seed)
    p_gross: list[float] = []
    p_net: list[float] = []
    turns: list[float] = []
    for _ in range(n_synthetic):
        work["_noise"] = ar1_noise_signal(months, keys, phi, rng)
        s = build_series(work, "_noise", direction, mode, spec)
        seed_i = int(rng.integers(0, 2**31 - 1))
        p_gross.append(block_bootstrap_p(np.asarray(s.gross), mean_block=mean_block,
                                         R=R, seed=seed_i)["p"])
        p_net.append(block_bootstrap_p(np.asarray(s.net), mean_block=mean_block,
                                       R=R, seed=seed_i)["p"])
        turns.append(float(np.mean(s.turnover)) if s.turnover else float("nan"))
    g = np.asarray(p_gross, dtype=np.float64)
    nt = np.asarray(p_net, dtype=np.float64)
    share_g = float((g <= alpha).mean()) if g.size else float("nan")
    share_n = float((nt <= alpha).mean()) if nt.size else float("nan")
    share = share_g if calibration_arm == "gross" else share_n
    return {
        "n_synthetic": int(n_synthetic), "phi": float(phi), "R": int(R),
        "alpha": float(alpha), "mean_block": float(mean_block),
        "calibration_arm": calibration_arm,
        "share_at_or_below_alpha": share,
        "share_gross": share_g, "share_net": share_n,
        "accept_band": [float(accept[0]), float(accept[1])],
        "pass": control_verdict(share, accept),
        "p_gross_median": float(np.median(g)) if g.size else float("nan"),
        "p_net_median": float(np.median(nt)) if nt.size else float("nan"),
        "mean_turnover": float(np.nanmean(turns)) if turns else float("nan"),
    }


# -------------------------------------------------------------------- verdicts
def verdict(p: float, mean: float, bar: float, suggestive_bar: float = 0.05) -> str:
    """SOMETHING at p <= bar with a positive mean; SUGGESTIVE in (bar, 0.05]; else NOTHING."""
    if not np.isfinite(p) or not np.isfinite(mean):
        return "INSTRUMENT-INVALID"
    if p <= bar and mean > 0:
        return "SOMETHING"
    if p <= suggestive_bar and mean > 0:
        return "SUGGESTIVE"
    return "NOTHING"


def effective_n(corr: np.ndarray) -> float:
    """N_eff = N^2 / sum of pairwise rho (the batch's effective independent count)."""
    c = np.asarray(corr, dtype=np.float64)
    n = c.shape[0]
    s = float(np.nansum(c))
    if s <= 0:
        return float("nan")
    return float(n * n / s)


def months_to_timestamps(month: pd.Series) -> pd.Series:
    """
    The panel's `month` column holds pandas Period ordinals (monthly), not
    timestamps. Converting with pd.to_datetime would read them as nanoseconds
    and silently select nothing, so the conversion is explicit here.
    """
    if pd.api.types.is_integer_dtype(month):
        return pd.PeriodIndex(
            [pd.Period(ordinal=int(o), freq="M") for o in month], freq="M"
        ).to_timestamp(how="end").normalize().to_series(index=month.index)
    return pd.to_datetime(month)


def subperiod_means(frame: pd.DataFrame, spans: list[tuple[str, str, str]],
                    col: str = "net") -> dict[str, dict]:
    out: dict[str, dict] = {}
    m = months_to_timestamps(frame["month"])
    for label, lo, hi in spans:
        sel = (m >= pd.Timestamp(lo)) & (m <= pd.Timestamp(hi))
        v = frame.loc[sel, col].to_numpy(dtype=np.float64)
        out[label] = {
            "months": int(v.size),
            "mean_bps": float(v.mean() * 1e4) if v.size else float("nan"),
        }
    return out
