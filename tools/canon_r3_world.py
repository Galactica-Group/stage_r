"""
================================================================================
Script:       canon_r3_world.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3
Purpose:      The scrambled worlds of Canon read 3, and the one door to the real
              one: a world's targets are a pure function of (panel, k), and
              world 0 can be built only from a verified registration object.
Entry Point:  No
Dependencies: numpy, pandas; tools/xs_triage.py
Last Sprint:  Sprint STOCK-1 — Canon read 3 built and controlled (2026-11-21)
================================================================================

World k >= 1 shifts each formation month's targets by one position along the
month's own key order, so the month's cross-sectional distribution of fwd_ret is
preserved exactly and the link between a name and its own target is destroyed.
Nothing moves in time: a target drawn in month m is always a month-m target.

World 0 is the real world. build_world refuses to return it on a flag: it
demands a VerifiedRegistration, which only canon_r3_gate.load_registration can
mint, and re-verifies that registration's file sha256 before it hands anything
back. The plant refuses world 0 outright.
================================================================================
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "1.0.0"

WORLD_SEED_BASE = 7000          # world k draws from default_rng(7000 + k)
PLANT_SEED_BASE = 9000          # plant draws from default_rng(9000 + 100k + j)
N_WORLDS = 60

_MINT = object()                # only the gate holds a reference to this


class WorldZeroRefused(PermissionError):
    """World 0 was asked for without a verified registration."""


class PlantRefused(PermissionError):
    """A plant was asked for on the real world."""


@dataclass(frozen=True)
class VerifiedRegistration:
    """A registration file whose sha256 was checked against a passed --sha."""
    path: str
    sha256: str
    data: dict
    _mint: object = None

    def __post_init__(self):
        if self._mint is not _MINT:
            raise PermissionError(
                "a VerifiedRegistration can only be minted by the gate's "
                "load_registration; it cannot be constructed directly")


def _mint_registration(path, sha256, data) -> VerifiedRegistration:
    """Private to the gate."""
    return VerifiedRegistration(path=str(path), sha256=sha256, data=data, _mint=_MINT)


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def world_keys(all_permatickers, k: int) -> pd.Series:
    """One key per permaticker, drawn over the SORTED permaticker list."""
    if int(k) < 1:
        raise WorldZeroRefused("world_keys is defined for k >= 1 only; got %r" % k)
    names = np.array(sorted(int(x) for x in set(int(v) for v in all_permatickers)),
                     dtype=np.int64)
    rng = np.random.default_rng(WORLD_SEED_BASE + int(k))
    return pd.Series(rng.random(names.shape[0]), index=names)


def scramble_targets(frame: pd.DataFrame, k: int, all_permatickers=None,
                     month_col: str = "month", key_col: str = "permaticker",
                     cols: tuple[str, ...] = ("fwd_ret", "fwd_stopped")
                     ) -> pd.DataFrame:
    """
    Return a copy of `frame` whose `cols` are shifted by one along each month's
    key order. The frame must hold one row per (month, permaticker).
    """
    if int(k) < 1:
        raise WorldZeroRefused("scramble_targets is defined for k >= 1 only")
    if all_permatickers is None:
        all_permatickers = frame[key_col]
    keys = world_keys(all_permatickers, k)
    out = frame.copy()
    kvals = out[key_col].map(keys)
    if kvals.isna().any():
        raise KeyError("a permaticker in the frame has no world key")
    if not out.index.is_unique:
        raise ValueError("scramble_targets needs a unique index; reset it first")
    out["_k"] = kvals.to_numpy(dtype=np.float64)
    pieces = []
    for _, g in out.groupby(month_col, sort=True, observed=True):
        order = np.argsort(g["_k"].to_numpy(), kind="stable")
        rows = g.index.to_numpy()[order]
        rolled = np.roll(rows, -1)          # each takes the NEXT name's target
        pieces.append((rows, rolled))
    src = np.concatenate([p[1] for p in pieces])
    dst = np.concatenate([p[0] for p in pieces])
    for c in cols:
        out.loc[dst, c] = frame.loc[src, c].to_numpy()
    return out.drop(columns=["_k"])


def build_world(frame: pd.DataFrame, k: int, registration=None,
                all_permatickers=None, **kw) -> pd.DataFrame:
    """
    The one door. k >= 1 scrambles. k == 0 is the real world and is refused
    unless `registration` is a VerifiedRegistration whose file still hashes to
    the sha256 it was minted with.
    """
    k = int(k)
    if k == 0:
        if not isinstance(registration, VerifiedRegistration):
            raise WorldZeroRefused(
                "world 0 needs a VerifiedRegistration, not %r" % type(registration))
        p = Path(registration.path)
        if not p.exists():
            raise WorldZeroRefused("the registration file is gone: %s" % p)
        if _sha256_file(p) != registration.sha256:
            raise WorldZeroRefused("the registration file no longer matches its sha256")
        return frame.copy()
    return scramble_targets(frame, k, all_permatickers=all_permatickers, **kw)


def plant_feature(frame: pd.DataFrame, k: int, rho: float, rho_index: int,
                  feature: str = "PRC", month_col: str = "month",
                  fwd_col: str = "fwd_ret") -> pd.DataFrame:
    """
    Replace `feature` by the within-month rank scaling of
    rho * z_y + sqrt(1 - rho^2) * e, where z_y is the within-month normal score
    of this world's own fwd_ret. Refused on world 0.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import xs_triage as xt
    import canon_r3_features as F

    if int(k) < 1:
        raise PlantRefused("the plant is for scrambled worlds only; got k=%r" % k)
    rng = np.random.default_rng(PLANT_SEED_BASE + 100 * int(k) + int(rho_index))
    out = frame.copy()
    vals = np.full(len(out), np.nan)
    pos = {ix: i for i, ix in enumerate(out.index)}
    for _, g in out.groupby(month_col, sort=True, observed=True):
        y = g[fwd_col].to_numpy(dtype=np.float64)
        zy = xt.rank_to_normal_z(y, "HIGH")
        e = rng.standard_normal(len(g))
        v = rho * zy + np.sqrt(1.0 - rho * rho) * e
        for ix, x in zip(g.index, v):
            vals[pos[ix]] = x
    out[feature] = vals
    out = F.rank_scale_within_month(out, [feature], month_col=month_col)
    return out
