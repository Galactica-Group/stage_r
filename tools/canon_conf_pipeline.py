"""
================================================================================
Script:       canon_conf_pipeline.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      One pipeline taking the panel, the features, the list of cuts, the
              world number and the key seed base. Label, model, book and costs
              are read 3's, unchanged; only the window and the seed base are
              lifted from constants to arguments.
Entry Point:  No
Dependencies: numpy, pandas; tools/canon_r3_machine.py, tools/canon_r3_features.py,
              tools/xs_triage.py
Last Sprint:  Sprint CONF-1 — the confirmation built and controlled (2026-11-23)
================================================================================

Two of read 3's functions fix their window by a module constant: walk_predict
reads cut_months() and build_book reads scored_months(). They are carried here
with the window lifted to a parameter and nothing else altered; the proof is
tripwire T3, which reproduces banked half-A worlds exactly through this code,
and a structural test that the lifted bodies differ from read 3's only in the
window lines.

World 0 is the real world. It is refused unless handed a VerifiedConfRegistration
that the confirmation's gate minted, and refused always on read 3's cuts: half A
is spent and is not rebuilt.
================================================================================
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon_r3_features as F3  # noqa: E402
import canon_r3_machine as M  # noqa: E402
import xs_triage as xt  # noqa: E402

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

VERSION = "1.0.0"

# ---- read 3's window, for tripwire T3 only
R3_FIRST_FORMATION = "1998-12"
R3_CUTS = M.cut_months()
R3_SCORED = M.scored_months()
R3_SEED_BASE = 7000

# ---- the confirmation's window
CONF_FIRST_FORMATION = "1998-12"
CONF_FIRST_CUT = "2012-12"
CONF_N_CUTS = 8
CONF_FIRST_SCORED = "2012-12"
CONF_LAST_SCORED = "2020-11"
CONF_N_SCORED_MONTHS = 96
CONF_SEED_BASE = 17000
CONF_PLANT_SEED_BASE = 19000
CONF_SEED_VOTING = 20261123

N_WORLDS = 60


def conf_cuts() -> list[int]:
    o = pd.Period(CONF_FIRST_CUT, freq="M").ordinal
    return [o + 12 * i for i in range(CONF_N_CUTS)]


def conf_scored() -> list[int]:
    a = pd.Period(CONF_FIRST_SCORED, freq="M").ordinal
    b = pd.Period(CONF_LAST_SCORED, freq="M").ordinal
    return list(range(a, b + 1))


CONF_CUTS = conf_cuts()
CONF_SCORED = conf_scored()


# --------------------------------------------------------------- the one door
_MINT = object()


class WorldZeroRefused(PermissionError):
    """World 0 was asked for without a verified registration, or on half A."""


class PlantRefused(PermissionError):
    """A plant was asked for on the real world."""


@dataclass(frozen=True)
class VerifiedConfRegistration:
    path: str
    sha256: str
    data: dict
    _mint: object = None

    def __post_init__(self):
        if self._mint is not _MINT:
            raise PermissionError(
                "a VerifiedConfRegistration can only be minted by the "
                "confirmation's gate; it cannot be constructed directly")


def _mint_registration(path, sha256, data) -> VerifiedConfRegistration:
    return VerifiedConfRegistration(path=str(path), sha256=sha256, data=data,
                                    _mint=_MINT)


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


# ------------------------------------------------------------- scrambled worlds
def world_keys(all_permatickers, k: int, seed_base: int) -> pd.Series:
    """One key per permaticker, drawn over the SORTED permaticker list."""
    if int(k) < 1:
        raise WorldZeroRefused("world_keys is defined for k >= 1 only; got %r" % k)
    names = np.array(sorted(int(x) for x in set(int(v) for v in all_permatickers)),
                     dtype=np.int64)
    rng = np.random.default_rng(int(seed_base) + int(k))
    return pd.Series(rng.random(names.shape[0]), index=names)


def scramble_targets(frame: pd.DataFrame, k: int, seed_base: int,
                     all_permatickers=None, month_col: str = "month",
                     key_col: str = "permaticker",
                     cols: tuple[str, ...] = ("fwd_ret", "fwd_stopped")
                     ) -> pd.DataFrame:
    """Each month's targets shifted one position along that month's key order."""
    if int(k) < 1:
        raise WorldZeroRefused("scramble_targets is defined for k >= 1 only")
    if all_permatickers is None:
        all_permatickers = frame[key_col]
    keys = world_keys(all_permatickers, k, seed_base)
    out = frame.copy()
    if not out.index.is_unique:
        raise ValueError("scramble_targets needs a unique index; reset it first")
    kvals = out[key_col].map(keys)
    if kvals.isna().any():
        raise KeyError("a permaticker in the frame has no world key")
    out["_k"] = kvals.to_numpy(dtype=np.float64)
    pieces = []
    for _, g in out.groupby(month_col, sort=True, observed=True):
        order = np.argsort(g["_k"].to_numpy(), kind="stable")
        rows = g.index.to_numpy()[order]
        pieces.append((rows, np.roll(rows, -1)))
    src = np.concatenate([p[1] for p in pieces])
    dst = np.concatenate([p[0] for p in pieces])
    present = [c for c in cols if c in frame.columns]
    for c in present:
        out.loc[dst, c] = frame.loc[src, c].to_numpy()
    return out.drop(columns=["_k"])


def build_world(frame: pd.DataFrame, k: int, cuts, registration=None,
                seed_base: int = CONF_SEED_BASE, all_permatickers=None,
                **kw) -> pd.DataFrame:
    """
    k >= 1 scrambles. k == 0 is the real world: refused unless `registration` is
    a VerifiedConfRegistration whose file still hashes to its recorded sha256,
    and refused ALWAYS on read 3's cuts, because half A is spent.
    """
    k = int(k)
    if k == 0:
        if list(cuts) == list(R3_CUTS):
            raise WorldZeroRefused(
                "world 0 is refused on read 3's cuts: half A is spent and is "
                "not rebuilt")
        if not isinstance(registration, VerifiedConfRegistration):
            raise WorldZeroRefused(
                "world 0 needs a VerifiedConfRegistration, not %r"
                % type(registration))
        p = Path(registration.path)
        if not p.exists():
            raise WorldZeroRefused("the registration file is gone: %s" % p)
        if _sha256_file(p) != registration.sha256:
            raise WorldZeroRefused(
                "the registration file no longer matches its sha256")
        return frame.copy()
    return scramble_targets(frame, k, seed_base,
                            all_permatickers=all_permatickers, **kw)


def plant_feature(frame: pd.DataFrame, k: int, rho: float, rho_index: int,
                  seed_base: int = CONF_PLANT_SEED_BASE, feature: str = "PRC",
                  month_col: str = "month", fwd_col: str = "fwd_ret"
                  ) -> pd.DataFrame:
    """Read 3's plant, with its own seed base. Refused on world 0."""
    if int(k) < 1:
        raise PlantRefused("the plant is for scrambled worlds only; got k=%r" % k)
    rng = np.random.default_rng(int(seed_base) + 100 * int(k) + int(rho_index))
    out = frame.copy()
    vals = np.full(len(out), np.nan)
    pos = {ix: i for i, ix in enumerate(out.index)}
    for _, g in out.groupby(month_col, sort=True, observed=True):
        zy = xt.rank_to_normal_z(g[fwd_col].to_numpy(dtype=np.float64), "HIGH")
        e = rng.standard_normal(len(g))
        v = rho * zy + np.sqrt(1.0 - rho * rho) * e
        for ix, x in zip(g.index, v):
            vals[pos[ix]] = x
    out[feature] = vals
    return F3.rank_scale_within_month(out, [feature], month_col=month_col)


# ------------------------------------- read 3's walk, the window as a parameter
def walk_predict(world: pd.DataFrame, y: np.ndarray, cuts,
                 first_formation: str = CONF_FIRST_FORMATION,
                 feature_names: list[str] | None = None,
                 month_col: str = "month") -> tuple[np.ndarray, list[dict]]:
    """
    Read 3's walk_predict with cut_months() lifted to `cuts` and
    F.FIRST_FORMATION_MONTH lifted to `first_formation`. Nothing else differs.
    """
    feature_names = list(F3.FEATURE_NAMES if feature_names is None else feature_names)
    X = world[feature_names].to_numpy(dtype=np.float64)
    m = world[month_col].to_numpy(dtype=np.int64)
    pred = np.full(len(world), np.nan)
    diags = []
    for C in cuts:
        tr = (m >= pd.Period(first_formation, freq="M").ordinal) & \
             (m <= C - 2) & np.isfinite(y)
        te = (m >= C) & (m <= C + 11)
        if tr.sum() < 1000 or te.sum() == 0:
            diags.append({"cut": str(pd.Period(ordinal=int(C), freq="M")),
                          "train_rows": int(tr.sum()), "score_rows": int(te.sum()),
                          "fit_seconds": 0.0, "skipped": True})
            continue
        t0 = time.time()
        acc = np.zeros(int(te.sum()))
        for s in M.SEEDS:
            mdl = HistGradientBoostingRegressor(random_state=s, **M.MODEL_KW)
            mdl.fit(X[tr], y[tr])
            acc += mdl.predict(X[te])
        pred[te] = acc / len(M.SEEDS)
        diags.append({"cut": str(pd.Period(ordinal=int(C), freq="M")),
                      "train_first": str(pd.Period(ordinal=int(m[tr].min()), freq="M")),
                      "train_last": str(pd.Period(ordinal=int(m[tr].max()), freq="M")),
                      "score_first": str(pd.Period(ordinal=int(m[te].min()), freq="M")),
                      "score_last": str(pd.Period(ordinal=int(m[te].max()), freq="M")),
                      "train_rows": int(tr.sum()), "score_rows": int(te.sum()),
                      "fit_seconds": round(time.time() - t0, 2), "skipped": False})
    return pred, diags


def build_book(world: pd.DataFrame, pred: np.ndarray, scored,
               fwd_col: str = "fwd_ret",
               month_col: str = "month") -> tuple[dict, dict]:
    """
    Read 3's build_book with scored_months() lifted to `scored`, and the realised
    return column lifted to `fwd_col`, which the registered series leaves at
    fwd_ret.
    """
    sm = set(int(x) for x in scored)
    w = world.copy()
    w["_pred"] = pred
    w = w.loc[w[month_col].isin(sm)].copy()
    s_base, diag = xt.build_series_v2(w, "_pred", "HIGH", "LS", xt.COST_BASE,
                                      fwd_col=fwd_col)
    s_stress, _ = xt.build_series_v2(w, "_pred", "HIGH", "LS", xt.COST_STRESS,
                                     fwd_col=fwd_col)
    fb, fs = s_base.frame(), s_stress.frame()
    if not fb["month"].equals(fs["month"]):
        raise AssertionError("the two cost specs produced different month sets")
    series = {
        "months": fb["month"].to_numpy(dtype=np.int64),
        "gross": fb["gross"].to_numpy(dtype=np.float64),
        "net_base": fb["net"].to_numpy(dtype=np.float64),
        "net_stress": fs["net"].to_numpy(dtype=np.float64),
        "turnover": fb["turnover"].to_numpy(dtype=np.float64),
    }
    return series, diag


# -------------------------------------------------------- one world, end to end
def run_world(features: pd.DataFrame, k: int, cuts, scored,
              seed_base: int = CONF_SEED_BASE, all_permatickers=None,
              stopped: bool = False, registration=None,
              plant_rho: float | None = None, plant_rho_index: int | None = None,
              first_formation: str = CONF_FIRST_FORMATION) -> dict:
    t0 = time.time()
    world = build_world(features, k, cuts, registration=registration,
                        seed_base=seed_base, all_permatickers=all_permatickers)
    if plant_rho is not None:
        world = plant_feature(world, k, plant_rho, plant_rho_index)
    y = M.make_label(world)
    pred, diags = walk_predict(world, y, cuts, first_formation=first_formation)
    series, bdiag = build_book(world, pred, scored)
    st = M.series_stats(series)
    out = {"k": int(k), "plant_rho": plant_rho,
           "series": {kk: series[kk].tolist() for kk in series},
           "stats": st, "book_diag": bdiag, "fit_diag": diags,
           "label_rows": int(np.isfinite(y).sum())}

    if stopped:
        # REPORTED, NEVER VOTED. Key 3 was struck by ruling (CONF-1b); what is
        # left is a bound, computed from the book's own weights by the harness's
        # own fm_slope_weights. It touches no label, no prediction and no key.
        import canon_conf_stopped as S
        tab = S.s1(world, pred, scored)
        out["stopped_bound"] = {
            "s1_summary": S.s1_summary(tab),
            "s1_by_month": tab.to_dict(orient="records"),
            "s2": S.s2(series["net_base"], series["months"], tab),
        }
    out["wall_seconds"] = round(time.time() - t0, 1)
    return out
