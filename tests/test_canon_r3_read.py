"""
================================================================================
Script:       test_canon_r3_read.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3
Purpose:      The STOCK-2 additions: the eight-value plant grid and its frozen
              seed indices, the tracked pool as the read's only source of the 60
              world statistics, the two measured floors, the two bars, and the
              reported-not-voted bundle.
Entry Point:  No — pytest
Dependencies: stdlib, numpy, pandas; tools/canon_r3_*.py, tools/run_canon_r3.py
Last Sprint:  Sprint STOCK-2 — Canon read 3: seal and read (2026-11-21)
================================================================================
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import canon_r3_machine as M  # noqa: E402
import canon_r3_paths as P  # noqa: E402
import canon_r3_synthetic as S  # noqa: E402
import run_canon_r3 as R  # noqa: E402


@pytest.fixture(autouse=True)
def synthetic(tmp_path):
    st = S.snapshot_state()
    P.set_roots(repo=tmp_path, snapshot=tmp_path, banked_panel=tmp_path / "n.parquet",
                derived=tmp_path / "d", scratch=tmp_path / "s",
                windows=tmp_path / "w.json")
    (tmp_path / "d").mkdir(exist_ok=True)
    (tmp_path / "s").mkdir(exist_ok=True)
    yield tmp_path
    S.restore_state(st)


# ------------------------------------------------------------ 2.1 the grid
def test_the_plant_grid_has_eight_values_in_the_registered_order():
    assert len(R.PLANT_GRID) == 8
    assert sorted(R.PLANT_RHOS) == [0.005, 0.010, 0.020, 0.025, 0.030, 0.035,
                                    0.040, 0.080]
    assert len(set(R.PLANT_RHOS)) == 8


def test_the_five_banked_rhos_keep_their_seed_indices():
    """The STOCK-1 rows must reproduce, so their seed indices are frozen."""
    want = {0.005: 0, 0.010: 1, 0.020: 2, 0.040: 3, 0.080: 4}
    got = dict(R.PLANT_GRID)
    for rho, j in want.items():
        assert got[rho] == j, (rho, got[rho], j)


def test_the_three_new_rhos_take_the_next_seed_indices():
    got = dict(R.PLANT_GRID)
    assert got[0.025] == 5 and got[0.030] == 6 and got[0.035] == 7


def test_every_seed_index_is_used_once():
    idx = [j for _, j in R.PLANT_GRID]
    assert sorted(idx) == list(range(8))


def test_the_plant_seed_differs_for_every_grid_cell():
    seeds = {(k, j): 9000 + 100 * k + j for k in R.PLANT_WORLDS
             for _, j in R.PLANT_GRID}
    assert len(set(seeds.values())) == len(seeds)


# ------------------------------------------------------ 2.3 the tracked pool
def _pool(n=60, seed=1):
    rng = np.random.default_rng(seed)
    ws = []
    for k in range(1, n + 1):
        ws.append({"k": k,
                   "gross": {"mean": float(rng.normal(0, 6e-4)), "sd": 0.0078,
                             "t": float(rng.normal(0, 1)), "sharpe": 0.07},
                   "net_base": {"mean": -0.0014, "sd": 0.0078, "t": -2.1,
                                "sharpe": -0.64},
                   "net_stress": {"mean": -0.004, "sd": 0.0078, "t": -5.0,
                                  "sharpe": -1.7},
                   "turnover_mean": 2.54, "months": 132, "label_rows": 165120})
    return ws


def test_the_bars_are_the_pool_median_and_the_third_largest_t():
    ws = _pool()
    b = R.bars(ws)
    gm = np.array([w["gross"]["mean"] for w in ws])
    gt = np.sort(np.array([w["gross"]["t"] for w in ws]))[::-1]
    assert abs(b["key1a_centre_pool_median_gross_mean"] - float(np.median(gm))) < 1e-15
    assert abs(b["key1b_third_largest_pool_gross_t"] - float(gt[2])) < 1e-15
    assert b["n_worlds"] == 60


def test_the_third_largest_t_is_exactly_the_key1b_boundary():
    """A read t just above it passes with 2 exceedances; just below fails with 3."""
    ws = _pool()
    gt = np.array([w["gross"]["t"] for w in ws])
    third = R.bars(ws)["key1b_third_largest_pool_gross_t"]
    assert M.key1b(third + 1e-9, gt)["exceedances"] == 2
    assert M.key1b(third + 1e-9, gt)["pass"] is True
    assert M.key1b(third - 1e-9, gt)["exceedances"] == 3
    assert M.key1b(third - 1e-9, gt)["pass"] is False


def test_the_read_takes_the_pool_from_the_tracked_file_and_nowhere_else():
    """Structural: stage_observed must read P.pool_v1() and not P.pool_manifest()."""
    src = (ROOT / "tools" / "run_canon_r3.py").read_text()
    fn = src[src.index("def stage_observed"):src.index("def append_ledger_row")]
    assert "P.pool_v1()" in fn
    assert "pool_manifest" not in fn, "the read must not reach the scratch manifest"
    assert "pool_dir" not in fn


def test_the_tracked_pool_writer_records_every_registered_statistic():
    src = (ROOT / "tools" / "run_canon_r3.py").read_text()
    fn = src[src.index("def write_pool_v1"):src.index("# ------------------------------------------------------------------ the stages")]
    for k in ("mean", "sd", "t", "sharpe"):
        assert '"%s"' % k in fn
    for leg in ("gross", "net_base", "net_stress"):
        assert '"%s":' % leg in fn
    assert "turnover_mean" in fn
    assert "scratch_manifest_sha256" in fn
    assert "wall_seconds" not in fn, \
        "the tracked pool must carry no timing, or it is not byte-reproducible"


# ------------------------------------------------------------- 3.3 the floors
def _plants(pass_from_rho, surv_from_rho):
    by = {}
    for rho in sorted(R.PLANT_RHOS):
        n1 = 12 if rho >= pass_from_rho else 0
        ns = 12 if rho >= surv_from_rho else 0
        by["%.3f" % rho] = {"rho": rho, "n_worlds": 12, "n_pass_key1": n1,
                            "verdict_counts": {"SURVIVOR": ns},
                            "median_gross_sharpe": rho * 80.0,
                            "median_net_base_sharpe": rho * 70.0}
    return {"by_rho": by}


def test_the_two_floors_are_the_smallest_rho_meeting_each_rule():
    f = R.floors(_plants(0.030, 0.040))
    assert f["key1_floor"]["rho"] == 0.030 and f["key1_floor"]["detected"]
    assert f["whole_verdict_floor"]["rho"] == 0.040
    assert f["key1_floor"]["median_gross_sharpe"] == pytest.approx(2.4)
    assert f["whole_verdict_floor"]["median_net_base_sharpe"] == pytest.approx(2.8)
    assert f["key2_sharpe_bar"] == 0.5
    assert f["grid"] == sorted(R.PLANT_RHOS)


def test_a_floor_not_reached_on_the_grid_is_reported_as_undetected():
    f = R.floors(_plants(99.0, 99.0))
    assert f["key1_floor"]["detected"] is False
    assert f["key1_floor"]["rho"] is None
    assert f["whole_verdict_floor"]["detected"] is False


def test_the_whole_verdict_floor_is_never_below_the_key1_floor_by_construction():
    f = R.floors(_plants(0.020, 0.035))
    assert f["key1_floor"]["rho"] <= f["whole_verdict_floor"]["rho"]


# --------------------------------------------------- 2.4 reported, not voted
def test_the_break_even_cost_is_the_gross_mean_over_the_traded_notional():
    assert M.break_even_bps_per_side(0.00127, 2.54) == pytest.approx(5.0)
    assert M.break_even_bps_per_side(-0.001, 2.0) == pytest.approx(-5.0)
    assert np.isnan(M.break_even_bps_per_side(0.001, 0.0))
    assert np.isnan(M.break_even_bps_per_side(float("nan"), 2.0))


def test_the_break_even_cost_is_the_cost_that_exactly_eats_the_gross_mean():
    import xs_triage as xt
    gross_mean, turn = 0.0019, 2.3
    be = M.break_even_bps_per_side(gross_mean, turn)
    spec = xt.CostSpec(label="BE", bps_per_side=be, borrow_bps_per_year=0.0)
    cost = ((spec.bps_per_side + spec.commission_bps) / 1e4) * turn
    assert cost == pytest.approx(gross_mean)


def test_the_yearly_table_covers_every_scored_month_once():
    months = np.array(M.scored_months(), dtype=np.int64)
    g = np.full(months.size, 0.001)
    n = np.full(months.size, -0.0005)
    rows = M.yearly_means(months, g, n)
    assert sum(r["months"] for r in rows) == 132
    assert rows[0]["year"] == 2001 and rows[-1]["year"] == 2012
    assert rows[0]["months"] == 1 and rows[-1]["months"] == 11
    for r in rows:
        assert r["gross_mean_bps"] == pytest.approx(10.0)
        assert r["net_base_mean_bps"] == pytest.approx(-5.0)


def test_the_net_against_zero_p_is_reported_and_centred_on_zero():
    rng = np.random.default_rng(7)
    net = rng.normal(-0.002, 0.008, 132)
    r = M.net_vs_zero_p(net, R=400, seed=3)
    assert 0.0 < r["p"] <= 1.0
    assert r["n"] == 132
    assert r["p"] > 0.5, "a negative-mean series must not look positive"


def test_the_full_report_carries_every_clause_two_four_item():
    months = np.array(M.scored_months(), dtype=np.int64)
    series = {"months": months,
              "gross": np.full(132, 0.001), "net_base": np.full(132, -0.0005),
              "net_stress": np.full(132, -0.002),
              "turnover": np.full(132, 2.5)}
    st = M.series_stats(series)
    bd = {"months_skipped_rank_deficient": 0, "months_skipped_too_small": 0,
          "cross_section_mean": 990.3, "cross_section_min": 983}
    rep = M.full_report(series, st, bd)
    for k in ("series_statistics", "turnover_mean", "break_even_bps_per_side",
              "net_base_vs_zero", "by_year", "months_scored",
              "months_skipped_rank_deficient", "months_skipped_too_small",
              "cross_section_mean", "cross_section_min"):
        assert k in rep, k
    assert set(rep["series_statistics"]) == {"gross", "net_base", "net_stress"}
    assert rep["months_scored"] == 132


# ------------------------------------------------------------- the ledger row
def test_the_ledger_constants_are_the_registered_ones():
    assert R.LEDGER_BATCH_ID == "CANON-R3"
    assert R.LEDGER_SIGNAL == "MACHINE_HGB_25PV"
    assert R.LEDGER_UNIVERSE == "top1000_tight_v2_halfA_132mo"
    assert R.LEDGER_VINTAGE == "20261004p"


def test_the_ledger_row_has_the_eleven_data_fields_the_chain_expects():
    import csv
    with open(ROOT / "reports" / "triage_ledger.csv", newline="") as fh:
        header = next(csv.reader(fh))
    assert len(header) == 13            # 11 data + prev_row_sha + row_sha
    src = (ROOT / "tools" / "run_canon_r3.py").read_text()
    fn = src[src.index("def append_ledger_row"):]
    assert "lc.verify" in fn and "lc.row_sha" in fn
    assert fn.count("lc.verify") == 2, "verify before AND after the append"


def test_the_pool_manifest_version_is_frozen_apart_from_the_tool_version():
    assert R.POOL_MANIFEST_VERSION == "1.0.0"
    assert R.VERSION != R.POOL_MANIFEST_VERSION
