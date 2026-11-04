"""
================================================================================
Script:       test_edgar_item408.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Fixture-only battery on tools/edgar_item408_probe.py: full-text
              search parsing and paging, identifier and URL construction,
              Inline XBRL ecd:* extraction and per-context grouping, flag and
              date resolution, manifest idempotency, the 8-per-second fair-access
              limiter on a fake clock, contact-string hygiene, and the
              containment guards.
Entry Point:  No — pytest
Dependencies: stdlib, pytest; the probe module via importlib
Last Sprint:  Sprint R5-0 — Item 408 event table and inventory (2026-11-04)
================================================================================

No network. Every answer here is hand-checkable from the fixture above it.

The mutations this battery is built to catch are: a flag resolver that guesses
instead of recording UNRESOLVED; an extractor that flattens two individuals in
two XBRL contexts into one row, or splits one into two; paging arithmetic that
walks past the endpoint's result ceiling or stops one page early; an archive URL
built from a zero-padded CIK; a fetch that re-downloads what the manifest
already banks; a limiter that lets a burst through; and a log line that carries
the SEC contact string instead of its length.
================================================================================
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "edgar_item408_probe.py"
ALLOWED_THIRD_PARTY = {"requests", "pandas", "pyarrow"}
FACTORY_DIR_NAME = "Monte_Carlo_Strategies"


def _module():
    spec = importlib.util.spec_from_file_location("edgar_item408_probe", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _module()


# ---------------------------------------------------------------------------
# Identifiers and URLs
# ---------------------------------------------------------------------------

def test_adsh_normalisation_accepts_both_spellings(m):
    assert m.normalise_adsh("0001437749-24-014820") == "0001437749-24-014820"
    assert m.normalise_adsh("000143774924014820") == "0001437749-24-014820"
    assert m.adsh_nodash("0001437749-24-014820") == "000143774924014820"


def test_adsh_normalisation_rejects_the_wrong_digit_count(m):
    for bad in ("0001437749-24-01482", "123", "0001437749-24-0148201"):
        with pytest.raises(ValueError):
            m.normalise_adsh(bad)


def test_archive_url_uses_the_integer_cik_and_the_bare_accession(m):
    url = m.build_archive_url("0000763532", "0001437749-24-014820", "lyts-20240331.htm")
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/763532/"
        "000143774924014820/lyts-20240331.htm"
    )
    # A bare integer CIK and a leading slash on the filename give the same URL.
    assert m.build_archive_url(763532, "000143774924014820", "/lyts-20240331.htm") == url


def test_archive_url_requires_a_filename(m):
    with pytest.raises(ValueError):
        m.build_archive_url("0000763532", "0001437749-24-014820", "   ")


def test_cik_from_secfilings_present_absent_and_malformed(m):
    present = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001027077"
    assert m.cik_from_secfilings(present) == "0001027077"
    # Short CIKs are zero-padded to ten digits.
    assert m.cik_from_secfilings("...&CIK=320193") == "0000320193"
    assert m.cik_from_secfilings(None) is None
    assert m.cik_from_secfilings("") is None
    assert m.cik_from_secfilings("https://www.sec.gov/cgi-bin/browse-edgar?action=x") is None
    assert m.cik_from_secfilings("CIK=notanumber") is None


# ---------------------------------------------------------------------------
# Full-text search: parsing, paging, month windows
# ---------------------------------------------------------------------------

def _fts_payload(total, ids):
    return {
        "hits": {
            "total": {"value": total, "relation": "eq"},
            "hits": [
                {
                    "_id": i,
                    "_source": {
                        "adsh": i.split(":")[0],
                        "ciks": ["0000763532"],
                        "form": "10-Q",
                        "file_type": "10-Q" if "ex_" not in i else "EX-19.1",
                        "file_date": "2024-05-06",
                        "period_ending": "2024-03-31",
                        "display_names": ["LSI INDUSTRIES INC  (LYTS)  (CIK 0000763532)"],
                    },
                }
                for i in ids
            ],
        }
    }


def test_parse_fts_page_splits_the_id_into_accession_and_filename(m):
    total, hits = m.parse_fts_page(
        _fts_payload(2877, ["0001437749-24-014820:lyts-20240331.htm",
                            "0001437749-24-014820:ex_664592.htm"])
    )
    assert total == 2877
    assert [h["adsh"] for h in hits] == ["0001437749-24-014820"] * 2
    assert [h["filename"] for h in hits] == ["lyts-20240331.htm", "ex_664592.htm"]
    assert hits[0]["cik"] == "0000763532"
    assert hits[0]["file_date"] == "2024-05-06"


def test_parse_fts_page_handles_an_empty_response(m):
    assert m.parse_fts_page({}) == (0, [])
    assert m.parse_fts_page({"hits": {"total": {"value": 0}, "hits": []}}) == (0, [])


def test_keep_primary_sets_aside_exhibits_and_amendments_and_counts_them(m):
    _t, hits = m.parse_fts_page(
        _fts_payload(3, ["0001437749-24-014820:lyts-20240331.htm",
                         "0001437749-24-014820:ex_664592.htm"])
    )
    primary, aside = m.keep_primary(hits)
    assert [h["filename"] for h in primary] == ["lyts-20240331.htm"]
    assert dict(aside) == {"EX-19.1": 1}


def test_page_offsets_arithmetic_and_the_result_ceiling(m):
    assert m.page_offsets(0) == []
    assert m.page_offsets(1) == [0]
    assert m.page_offsets(100) == [0]
    assert m.page_offsets(101) == [0, 100]
    assert m.page_offsets(250) == [0, 100, 200]
    # The endpoint cannot be paged past its from+size ceiling.
    assert m.page_offsets(25000)[-1] == 9900
    assert len(m.page_offsets(25000)) == 100


def test_month_windows_are_inclusive_and_clipped_at_both_ends(m):
    assert m.month_windows("2023-04-01", "2023-06-15") == [
        ("2023-04-01", "2023-04-30"),
        ("2023-05-01", "2023-05-31"),
        ("2023-06-01", "2023-06-15"),
    ]
    # A mid-month start is not back-filled to the first of the month.
    assert m.month_windows("2024-02-10", "2024-02-20") == [("2024-02-10", "2024-02-20")]
    # February 2024 is a leap month; the window must end on the 29th.
    assert m.month_windows("2024-02-01", "2024-03-01")[0] == ("2024-02-01", "2024-02-29")
    assert len(m.month_windows("2023-04-01", "2026-10-31")) == 43
    with pytest.raises(ValueError):
        m.month_windows("2024-05-01", "2024-04-01")


def test_fts_params_carry_the_custom_date_range(m):
    p = m.fts_params('"10b5-1" terminated', ["10-Q", "10-K"], "2024-05-01", "2024-05-31", 200)
    assert p["q"] == '"10b5-1" terminated'
    assert p["forms"] == "10-Q,10-K"
    assert p["dateRange"] == "custom"
    assert (p["startdt"], p["enddt"]) == ("2024-05-01", "2024-05-31")
    assert p["from"] == 200 and p["size"] == 100


# ---------------------------------------------------------------------------
# Flag resolution
# ---------------------------------------------------------------------------

def test_flag_resolution_prefers_the_ixt_format(m):
    assert m.resolve_flag("ixt:fixed-true", "") is True
    assert m.resolve_flag("ixt:fixed-false", "") is False
    assert m.resolve_flag("ixt-sec:booleantrue", "") is True
    assert m.resolve_flag("ixt-sec:booleanfalse", "") is False
    # The format wins even when the text would say otherwise.
    assert m.resolve_flag("ixt:fixed-false", "true") is False


def test_flag_resolution_falls_back_to_text_including_ballot_boxes(m):
    for t in ("true", "TRUE", " Yes ", "yes", "X", "☒", "☑", "[x]"):
        assert m.resolve_flag(None, t) is True, t
    for t in ("false", "No", "n", "☐", "[ ]"):
        assert m.resolve_flag(None, t) is False, t


def test_flag_resolution_records_unresolved_rather_than_guessing(m):
    for t in (None, "", "   ", "see below", "N/A", "--", "—"):
        assert m.resolve_flag(None, t) == m.UNRESOLVED, t
    # An unrecognised format with unrecognised text is UNRESOLVED, not False.
    assert m.resolve_flag("ixt:numdotdecimal", "1,000") == m.UNRESOLVED


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def test_date_parsing_covers_the_ixt_families_in_use(m):
    assert m.parse_ix_date("ixt:date-monthname-day-year-en", "May 13, 2024") == "2024-05-13"
    assert m.parse_ix_date("ixt:date-monthname-day-year-en", "Sept. 1, 2023") is None
    assert m.parse_ix_date("ixt:date-monthname-day-year-en", "Sep 1, 2023") == "2023-09-01"
    assert m.parse_ix_date("ixt:date-day-monthname-year-en", "3 March 2025") == "2025-03-03"
    assert m.parse_ix_date("ixt:date-month-day-year", "5/13/2024") == "2024-05-13"
    assert m.parse_ix_date("ixt:date-month-day-year", "5-13-24") == "2024-05-13"
    assert m.parse_ix_date(None, "2024-05-13") == "2024-05-13"
    assert m.parse_ix_date("ixt:date-monthname-day-year-en", "February 29th, 2024") == "2024-02-29"


def test_date_parsing_returns_none_on_an_impossible_or_absent_date(m):
    assert m.parse_ix_date(None, "") is None
    assert m.parse_ix_date(None, None) is None
    assert m.parse_ix_date(None, "February 30, 2024") is None
    assert m.parse_ix_date(None, "13/13/2024") is None
    assert m.parse_ix_date(None, "not a date") is None


# ---------------------------------------------------------------------------
# Inline XBRL extraction
# ---------------------------------------------------------------------------

TWO_INDIVIDUALS = """
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>
<p>Item 408(a). Trading Arrangements.</p>
<table>
 <tr>
  <td><ix:nonNumeric contextRef="c-71" name="ecd:TrdArrIndName">Jane Roe</ix:nonNumeric></td>
  <td><ix:nonNumeric contextRef="c-71" name="ecd:TrdArrIndTitle">Chief Financial <span>Officer</span></ix:nonNumeric></td>
  <td><ix:nonNumeric contextRef="c-71" name="ecd:TrdArrTerminatedFlag" format="ixt-sec:boolballotbox">&#9746;</ix:nonNumeric></td>
  <td><ix:nonNumeric contextRef="c-71" name="ecd:TrdArrTerminationDate" format="ixt:date-monthname-day-year-en">May 13, 2024</ix:nonNumeric></td>
  <td><ix:nonFraction contextRef="c-71" name="ecd:TrdArrSecuritiesAggAvailAmt" decimals="0">12,000</ix:nonFraction></td>
  <td><ix:nonNumeric contextRef="c-71" name="ecd:Rule10b51ArrTrmntdFlag" format="ixt:fixed-true">true</ix:nonNumeric></td>
 </tr>
 <tr>
  <td><ix:nonNumeric contextRef="c-72" name="ecd:TrdArrIndName">John Doe</ix:nonNumeric></td>
  <td><ix:nonNumeric contextRef="c-72" name="ecd:TrdArrIndTitle">Director</ix:nonNumeric></td>
  <td><ix:nonNumeric contextRef="c-72" name="ecd:TrdArrAdoptedFlag" format="ixt:fixed-true">true</ix:nonNumeric></td>
  <td><ix:nonNumeric contextRef="c-72" name="ecd:TrdArrAdoptionDate" format="ixt:date-monthname-day-year-en">March 4, 2024</ix:nonNumeric></td>
  <td><ix:nonNumeric contextRef="c-72" name="ecd:TrdArrDuration">180 days</ix:nonNumeric></td>
 </tr>
</table>
<p>Not a tagged paragraph.</p>
</body></html>
"""


def test_extraction_finds_every_ecd_name_and_only_those(m):
    names = m.ecd_names(TWO_INDIVIDUALS)
    assert names == [
        "ecd:Rule10b51ArrTrmntdFlag",
        "ecd:TrdArrAdoptedFlag",
        "ecd:TrdArrAdoptionDate",
        "ecd:TrdArrDuration",
        "ecd:TrdArrIndName",
        "ecd:TrdArrIndTitle",
        "ecd:TrdArrSecuritiesAggAvailAmt",
        "ecd:TrdArrTerminatedFlag",
        "ecd:TrdArrTerminationDate",
    ]


def test_extraction_groups_two_individuals_into_exactly_two_rows_by_context(m):
    rows = m.facts_to_rows(m.extract_ecd_facts(TWO_INDIVIDUALS))
    assert len(rows) == 2
    assert [r["context_ref"] for r in rows] == ["c-71", "c-72"]

    a, b = rows
    # Text spanning a nested element is joined, not truncated at the span.
    assert a["ecd:TrdArrIndName"] == "Jane Roe"
    assert a["ecd:TrdArrIndTitle"] == "Chief Financial Officer"
    assert a["ecd:TrdArrTerminatedFlag"] is True
    assert a["ecd:TrdArrTerminationDate"] == "2024-05-13"
    assert a["ecd:TrdArrTerminationDate_raw"] == "May 13, 2024"
    assert a["ecd:TrdArrSecuritiesAggAvailAmt"] == "12,000"
    assert a["ecd:Rule10b51ArrTrmntdFlag"] is True
    # The terminated individual carries no adoption fields at all.
    assert "ecd:TrdArrAdoptedFlag" not in a

    assert b["ecd:TrdArrIndName"] == "John Doe"
    assert b["ecd:TrdArrAdoptedFlag"] is True
    assert b["ecd:TrdArrAdoptionDate"] == "2024-03-04"
    assert b["ecd:TrdArrDuration"] == "180 days"
    assert "ecd:TrdArrTerminatedFlag" not in b


def test_extraction_keeps_the_enclosing_block_text_verbatim_and_classifies_nothing(m):
    rows = m.facts_to_rows(m.extract_ecd_facts(TWO_INDIVIDUALS))
    raw = rows[0]["container_text"]
    assert "Jane Roe" in raw and "May 13, 2024" in raw
    # The block is the table the facts sit in, not the untagged paragraph after it.
    assert "Not a tagged paragraph" not in raw


def test_a_fact_inside_a_paragraph_takes_the_paragraph_not_the_outer_table(m):
    html = (
        "<table><tr><td>" + ("filler " * 40) +
        '<p>Short block <ix:nonNumeric contextRef="c-1" '
        'name="ecd:TrdArrIndName">Ann Lee</ix:nonNumeric></p>'
        "</td></tr></table>"
    )
    rows = m.facts_to_rows(m.extract_ecd_facts(html))
    assert len(rows) == 1
    assert rows[0]["container_text"] == "Short block Ann Lee"


NO_ECD = """
<html><body><p>The Company had no Rule 10b5-1 trading arrangements that were
terminated during the quarter.</p>
<ix:nonNumeric contextRef="c-1" name="dei:EntityCentralIndexKey">0000763532</ix:nonNumeric>
</body></html>
"""


def test_a_filing_with_no_ecd_tags_yields_zero_rows(m):
    assert m.ecd_names(NO_ECD) == []
    assert m.facts_to_rows(m.extract_ecd_facts(NO_ECD)) == []
    meta = {"adsh": "0001437749-24-014820", "cik": "0000763532", "form": "10-Q",
            "file_date": "2024-05-06", "period_ending": "2024-03-31"}
    assert m.event_rows_for_filing(meta, NO_ECD) == []


def test_event_rows_carry_the_filing_metadata_and_the_mapped_columns(m):
    meta = {"adsh": "0001437749-24-014820", "cik": "0000763532", "form": "10-Q",
            "file_date": "2024-05-06", "period_ending": "2024-03-31"}
    rows = m.event_rows_for_filing(meta, TWO_INDIVIDUALS)
    assert len(rows) == 2
    r = rows[0]
    assert set(r) == set(m.EVENT_COLUMNS)
    assert r["adsh"] == meta["adsh"] and r["cik"] == "0000763532"
    assert r["individual_name"] == "Jane Roe"
    assert r["individual_title"] == "Chief Financial Officer"
    assert r["terminated_flag"] is True
    assert r["termination_date"] == "2024-05-13"
    assert r["termination_date_raw"] == "May 13, 2024"
    assert r["rule10b51_terminated_flag"] is True
    assert r["securities_agg_available"] == "12,000"
    assert r["adopted_flag"] is None
    assert "Jane Roe" in r["raw_text"]


def test_an_unresolvable_flag_reaches_the_row_as_unresolved(m):
    html = (
        '<table><tr><td><ix:nonNumeric contextRef="c-9" name="ecd:TrdArrIndName">'
        'Sam Poe</ix:nonNumeric></td>'
        '<td><ix:nonNumeric contextRef="c-9" name="ecd:TrdArrTerminatedFlag">'
        '—</ix:nonNumeric></td></tr></table>'
    )
    meta = {"adsh": "0001437749-24-014820", "cik": "0000763532", "form": "10-Q",
            "file_date": "2024-05-06", "period_ending": "2024-03-31"}
    rows = m.event_rows_for_filing(meta, html)
    assert rows[0]["terminated_flag"] == m.UNRESOLVED


# ---------------------------------------------------------------------------
# Disclosure lag (a date difference, not a price statistic)
# ---------------------------------------------------------------------------

def test_disclosure_lag_is_calendar_days_and_none_when_undated(m):
    assert m.disclosure_lag_days("2024-05-06", "2024-05-13") == -7
    assert m.disclosure_lag_days("2024-05-06", "2024-02-06") == 90
    assert m.disclosure_lag_days("2024-05-06", None) is None
    assert m.disclosure_lag_days(None, "2024-05-13") is None
    assert m.disclosure_lag_days("2024-05-06", "not a date") is None


def test_title_mix_is_a_substring_count_not_a_classification(m):
    mix = m.title_mix([
        "Chief Financial Officer",   # Chief and Officer both
        "Director",
        "President and CEO",
        "Trustee of the Trust",
        "Vice Chair",               # none of the buckets
        None,
    ])
    assert mix["Chief"] == 1
    assert mix["Officer"] == 1
    assert mix["Director"] == 1
    assert mix["President"] == 1
    assert mix["Trust"] == 1
    assert mix["other"] == 2


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_round_trips_and_skips_what_is_already_banked(m, tmp_path):
    p = tmp_path / "sub" / "manifest.json"
    assert m.load_manifest(str(p)) == {}
    man = {"vintage": "20261104", "docs": {"0001437749-24-014820": {"bytes": 10}}}
    m.write_manifest(str(p), man)
    back = m.load_manifest(str(p))
    assert back == man
    assert json.loads(p.read_text()) == man
    assert m.already_fetched(back, "0001437749-24-014820") is True
    assert m.already_fetched(back, "0000000000-00-000000") is False
    assert m.already_fetched({}, "0001437749-24-014820") is False


def test_vintage_paths_are_rooted_in_this_repo(m, tmp_path):
    d = m.vintage_dir("20261104", root=str(tmp_path))
    assert d == str(tmp_path / "data" / "edgar_item408" / "20261104")
    assert m.docs_dir("20261104", root=str(tmp_path)).endswith("/docs")
    assert m.tracked_manifest_path("20261104", root=str(tmp_path)) == str(
        tmp_path / "manifests" / "edgar_item408_20261104.json"
    )


# ---------------------------------------------------------------------------
# Fair access: the limiter, on a fake clock
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(float(seconds), 0.0)


def test_limiter_never_exceeds_eight_calls_in_any_one_second_window(m):
    clock = FakeClock()
    lim = m.RateLimiter(max_per_sec=8, time_fn=clock.time, sleep_fn=clock.sleep)
    stamps = [lim.acquire() for _ in range(200)]
    assert len(stamps) == 200
    assert stamps == sorted(stamps)
    for i, t0 in enumerate(stamps):
        inside = sum(1 for t in stamps[i:] if t - t0 < 1.0)
        assert inside <= 8, "window from %.6f held %d calls" % (t0, inside)
    # 200 calls at 8 per second cannot finish inside 24 seconds.
    assert stamps[-1] >= 24.0


def test_limiter_lets_a_burst_up_to_the_cap_through_without_sleeping(m):
    clock = FakeClock()
    lim = m.RateLimiter(max_per_sec=8, time_fn=clock.time, sleep_fn=clock.sleep)
    stamps = [lim.acquire() for _ in range(8)]
    assert stamps == [0.0] * 8
    assert lim.acquire() >= 1.0


def test_a_limiter_with_a_higher_cap_would_fail_the_window_assertion(m):
    """Non-vacuity: the window check above bites on an over-permissive limiter."""
    clock = FakeClock()
    lim = m.RateLimiter(max_per_sec=20, time_fn=clock.time, sleep_fn=clock.sleep)
    stamps = [lim.acquire() for _ in range(40)]
    worst = max(
        sum(1 for t in stamps[i:] if t - t0 < 1.0) for i, t0 in enumerate(stamps)
    )
    assert worst > 8


# ---------------------------------------------------------------------------
# Contact string hygiene
# ---------------------------------------------------------------------------

def test_contact_is_read_from_the_environment_when_set(m, tmp_path):
    value, source = m.read_contact(env={"EDGAR_CONTACT": "Acme Research ops@acme.test"})
    assert value == "Acme Research ops@acme.test"
    assert source == "env:EDGAR_CONTACT"


def test_contact_falls_back_to_the_config_file_and_strips_the_newline(m, tmp_path):
    f = tmp_path / "edgar_contact"
    f.write_text("Acme Research ops@acme.test\n")
    value, source = m.read_contact(env={}, file_path=str(f))
    assert value == "Acme Research ops@acme.test"
    assert source.startswith("file:")


def test_contact_absent_from_both_sources_stops_the_run(m, tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(m.StopRun):
        m.read_contact(env={}, file_path=str(missing))
    empty = tmp_path / "empty"
    empty.write_text("")
    with pytest.raises(m.StopRun):
        m.read_contact(env={}, file_path=str(empty))


def test_contact_failing_the_shape_check_stops_the_run(m, tmp_path):
    f = tmp_path / "edgar_contact"
    f.write_text("noatsign here but long enough\n")
    with pytest.raises(m.StopRun):
        m.read_contact(env={}, file_path=str(f))
    f.write_text("a@b\n")
    with pytest.raises(m.StopRun):
        m.read_contact(env={}, file_path=str(f))


def test_no_log_line_the_tool_emits_carries_the_contact_string(m, monkeypatch, capsys):
    """Behavioural: drive the real command path and read everything it printed."""
    sentinel = "Sentinel Co leak-canary@sentinel.test"
    monkeypatch.setenv("EDGAR_CONTACT", sentinel)

    class Boom:
        def __init__(self, *a, **k):
            raise m.StopRun("constructed stop before any request")

    monkeypatch.setattr(m, "SecSession", Boom)
    rc = m.main(["count", "--vintage", "unit", "--variants", "2024-05"])
    out = capsys.readouterr().out
    assert rc == 3
    assert sentinel not in out
    assert "leak-canary@sentinel.test" not in out
    assert "byte length %d" % len(sentinel.encode()) in out


def _parents(tree):
    out = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


def _inside_len_call(node, parents):
    cur = parents.get(node)
    while cur is not None:
        if (isinstance(cur, ast.Call) and isinstance(cur.func, ast.Name)
                and cur.func.id == "len"):
            return True
        cur = parents.get(cur)
    return False


def test_every_contact_reference_in_a_log_call_is_only_its_length():
    """A log line may carry len(contact), never the contact value itself."""
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    parents = _parents(tree)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name not in ("say", "print"):
            continue
        for arg in ast.walk(node):
            if isinstance(arg, ast.Name) and arg.id in ("contact", "_contact"):
                if not _inside_len_call(arg, parents):
                    offenders.append(ast.unparse(node))
            if isinstance(arg, ast.Attribute) and arg.attr == "_contact":
                if not _inside_len_call(arg, parents):
                    offenders.append(ast.unparse(node))
    assert not offenders, "contact value reaches a log call: %s" % offenders


def test_the_log_guard_bites_on_a_constructed_leak():
    """Non-vacuity: the same walk flags a say() that prints the value."""
    leaky = ast.parse("say('contact is %s' % contact)")
    parents = _parents(leaky)
    found = [
        n for n in ast.walk(leaky)
        if isinstance(n, ast.Name) and n.id == "contact"
        and not _inside_len_call(n, parents)
    ]
    assert found, "the guard would not catch a direct leak"


def test_stop_messages_report_a_path_not_a_header(m):
    assert m.url_path("https://efts.sec.gov/LATEST/search-index?q=%22x%22") == (
        "/LATEST/search-index"
    )
    assert m.url_path("https://www.sec.gov/Archives/edgar/data/1/2/a.htm") == (
        "/Archives/edgar/data/1/2/a.htm"
    )


# ---------------------------------------------------------------------------
# Containment guards
# ---------------------------------------------------------------------------

def _top_level_imports(tree):
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
            elif node.level:
                mods.add("__RELATIVE__")
    return mods


def test_imports_are_stdlib_or_allowed_third_party():
    mods = _top_level_imports(ast.parse(TOOL.read_text(encoding="utf-8")))
    assert "__RELATIVE__" not in mods, "relative import found; the tool must be self-contained"
    stdlib = set(sys.stdlib_module_names)
    offenders = {m_ for m_ in mods if m_ not in stdlib and m_ not in ALLOWED_THIRD_PARTY}
    assert not offenders, "disallowed imports: %s" % sorted(offenders)


def test_the_tool_names_no_factory_path():
    offenders = [
        n.value for n in ast.walk(ast.parse(TOOL.read_text(encoding="utf-8")))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and FACTORY_DIR_NAME in n.value
    ]
    assert not offenders, "factory path literal(s): %s" % offenders


def _docstring_nodes(tree):
    """The Constant nodes that are module, class or function docstrings."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                out.add(id(body[0].value))
    return out


def test_the_tool_opens_no_dotenv_and_reads_no_credential():
    """Prose may say the tool opens no .env; the code may not name one.

    The scan is over executable string literals and identifiers only, so a
    docstring that explains the policy cannot satisfy or break it.
    """
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    docs = _docstring_nodes(tree)
    literals = [
        n.value.lower() for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and id(n) not in docs
    ]
    identifiers = [
        (n.id if isinstance(n, ast.Name) else n.attr).lower()
        for n in ast.walk(tree) if isinstance(n, (ast.Name, ast.Attribute))
    ]
    kwargs = [
        (k.arg or "").lower() for n in ast.walk(tree) if isinstance(n, ast.Call)
        for k in n.keywords
    ]
    # Filenames and credential names may not appear as executable literals.
    for token in (".env", "dotenv", "api_key", "apikey", "authorization",
                  "bearer", "secret", "password"):
        hits = [s for s in literals if token in s]
        assert not hits, "credential-shaped literal in the tool: %s -> %s" % (token, hits)
    # Credential names may not appear as identifiers either. ".env"/"dotenv" are
    # excluded here: "env" is the legitimate name of read_contact's environment
    # argument, and flattening ".env" to "env" would collide with it.
    for token in ("api_key", "apikey", "authorization", "bearer", "secret",
                  "password"):
        flat = token.replace("_", "")
        bad = [i for i in identifiers + kwargs if i.replace("_", "") == flat]
        assert not bad, "credential-shaped identifier in the tool: %s" % token
    assert "dotenv" not in identifiers


def test_the_credential_scan_bites_on_a_constructed_violation():
    """Non-vacuity: the scan flags a literal and an identifier outside a docstring."""
    tree = ast.parse('"""policy prose about .env"""\nx = ".env"\napi_key = 1\n')
    docs = _docstring_nodes(tree)
    literals = [
        n.value.lower() for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and id(n) not in docs
    ]
    identifiers = [n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)]
    assert any(".env" in s for s in literals)
    assert "api_key" in identifiers


def test_all_writes_are_rooted_in_this_repo():
    src = TOOL.read_text(encoding="utf-8")
    assert 'REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))' in src
    for marker in ('"data", "edgar_item408"', '"manifests"', '"logs", "r5"'):
        assert marker in src, "expected repo-rooted output path: %s" % marker


# ---------------------------------------------------------------------------
# Retry budget on a fake session
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.content = b"x" * 10
        self.url = "https://efts.sec.gov/LATEST/search-index?q=%22secret-ish%22"

    def json(self):
        return self._payload


class FakeSession:
    """Answers a scripted list of statuses, then 200 forever."""

    def __init__(self, statuses, payload=None):
        self.statuses = list(statuses)
        self.payload = payload or {"hits": {"total": {"value": 0}, "hits": []}}
        self.calls = 0
        self.headers_seen = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        self.headers_seen.append(headers or {})
        status = self.statuses.pop(0) if self.statuses else 200
        return FakeResponse(status, self.payload)


def _session(m, statuses):
    fake = FakeSession(statuses)
    clock = FakeClock()
    lim = m.RateLimiter(max_per_sec=8, time_fn=clock.time, sleep_fn=clock.sleep)
    sess = m.SecSession("Acme Research ops@acme.test", limiter=lim,
                        session=fake, sleep_fn=clock.sleep)
    return sess, fake


def test_a_throttled_request_is_retried_and_then_succeeds(m):
    sess, fake = _session(m, [429, 403, 503])
    out = sess.get("https://efts.sec.gov/LATEST/search-index", expect_json=True)
    assert out == {"hits": {"total": {"value": 0}, "hits": []}}
    assert fake.calls == 4, "three retries then the success"
    assert sess.calls == 4


def test_a_transient_five_hundred_is_retried_not_fatal(m):
    sess, fake = _session(m, [500])
    sess.get("https://efts.sec.gov/LATEST/search-index", expect_json=True)
    assert fake.calls == 2


def test_the_retry_budget_is_three_and_then_the_run_stops(m):
    sess, fake = _session(m, [429, 429, 429, 429])
    with pytest.raises(m.StopRun) as exc:
        sess.get("https://efts.sec.gov/LATEST/search-index", expect_json=True)
    assert fake.calls == 4, "the fourth failure is not retried again"
    msg = str(exc.value)
    assert "429" in msg
    assert "/LATEST/search-index" in msg
    # The message carries a path, never a header or a query string.
    assert "User-Agent" not in msg and "ops@acme.test" not in msg and "q=" not in msg


def test_a_non_transient_status_is_not_retried(m):
    sess, fake = _session(m, [404])
    with pytest.raises(m.StopRun):
        sess.get("https://www.sec.gov/Archives/edgar/data/1/2/a.htm")
    assert fake.calls == 1


def test_every_request_sends_the_contact_user_agent_and_accepts_gzip(m):
    sess, fake = _session(m, [])
    sess.get("https://efts.sec.gov/LATEST/search-index", expect_json=True)
    h = fake.headers_seen[0]
    assert h["User-Agent"] == "Acme Research ops@acme.test"
    assert "gzip" in h["Accept-Encoding"]


def test_requests_pass_through_the_limiter(m):
    """Fair access is not optional: the limiter is consulted on every call."""
    fake = FakeSession([])
    clock = FakeClock()

    class CountingLimiter(m.RateLimiter):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.acquired = 0

        def acquire(self):
            self.acquired += 1
            return super().acquire()

    lim = CountingLimiter(max_per_sec=8, time_fn=clock.time, sleep_fn=clock.sleep)
    sess = m.SecSession("Acme Research ops@acme.test", limiter=lim,
                        session=fake, sleep_fn=clock.sleep)
    for _ in range(12):
        sess.get("https://efts.sec.gov/LATEST/search-index", expect_json=True)
    assert lim.acquired == 12
    assert clock.t >= 1.0, "the twelfth call cannot land in the first second"


# ---------------------------------------------------------------------------
# Month-level resume
# ---------------------------------------------------------------------------

def _count_args(m, tmp_path, start, end, template='"10b5-1" {term}'):
    import argparse
    return argparse.Namespace(
        vintage="unit", start=start, end=end, variants=None,
        and_template=template, union_cap=12000, func=m.cmd_count,
    )


def _drive_count(m, monkeypatch, tmp_path, start, end, template='"10b5-1" {term}'):
    """Run cmd_count against a fake endpoint rooted in tmp_path."""
    monkeypatch.setattr(m, "REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("EDGAR_CONTACT", "Acme Research ops@acme.test")
    months_hit = []

    class Fake:
        def __init__(self, *a, **k):
            self.calls = 0
            self.bytes = 0

        def get(self, url, params=None, expect_json=False):
            self.calls += 1
            months_hit.append(params["startdt"][:7])
            adsh = "00014377%02d-24-014820" % (len(months_hit) % 90)
            return {
                "hits": {
                    "total": {"value": 1, "relation": "eq"},
                    "hits": [{
                        "_id": "%s:doc.htm" % adsh,
                        "_source": {"ciks": ["0000763532"], "form": "10-Q",
                                    "file_type": "10-Q", "file_date": params["startdt"],
                                    "period_ending": None, "display_names": ["X"]},
                    }],
                }
            }

    monkeypatch.setattr(m, "SecSession", Fake)
    rc = m.cmd_count(_count_args(m, tmp_path, start, end, template))
    return rc, months_hit


def test_a_matching_checkpoint_is_resumed_and_counted_months_are_not_refetched(
        m, monkeypatch, tmp_path):
    rc, first = _drive_count(m, monkeypatch, tmp_path, "2023-04-01", "2023-05-31")
    assert rc == 0
    assert sorted(set(first)) == ["2023-04", "2023-05"]
    ck = json.loads((tmp_path / "data" / "edgar_item408" / "unit" / "union.json").read_text())
    assert [r["month"] for r in ck["per_month"]] == ["2023-04", "2023-05"]

    # Extend the range: only the new month is requested.
    rc, second = _drive_count(m, monkeypatch, tmp_path, "2023-04-01", "2023-06-30")
    assert rc == 0
    assert sorted(set(second)) == ["2023-06"], "already-counted months were refetched"
    ck2 = json.loads((tmp_path / "data" / "edgar_item408" / "unit" / "union.json").read_text())
    assert [r["month"] for r in ck2["per_month"]] == ["2023-04", "2023-05", "2023-06"]


def test_a_checkpoint_from_a_different_query_shape_is_discarded(m, monkeypatch, tmp_path):
    _drive_count(m, monkeypatch, tmp_path, "2023-04-01", "2023-05-31")
    rc, again = _drive_count(m, monkeypatch, tmp_path, "2023-04-01", "2023-05-31",
                             template='"10b5-1" AND {term}')
    assert rc == 0
    assert sorted(set(again)) == ["2023-04", "2023-05"], \
        "a changed query template must not reuse the old counts"


def test_the_union_cap_stops_the_run_before_any_document_fetch(m, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("EDGAR_CONTACT", "Acme Research ops@acme.test")

    class Fake:
        def __init__(self, *a, **k):
            self.calls = 0
            self.bytes = 0
            self._n = 0

        def get(self, url, params=None, expect_json=False):
            self._n += 1
            return {
                "hits": {
                    "total": {"value": 1},
                    "hits": [{
                        "_id": "%010d-24-014820:doc.htm" % self._n,
                        "_source": {"ciks": ["0000763532"], "form": "10-Q",
                                    "file_type": "10-Q", "file_date": params["startdt"],
                                    "period_ending": None, "display_names": ["X"]},
                    }],
                }
            }

    monkeypatch.setattr(m, "SecSession", Fake)
    args = _count_args(m, tmp_path, "2023-04-01", "2023-06-30")
    args.union_cap = 2
    with pytest.raises(m.StopRun) as exc:
        m.cmd_count(args)
    assert "exceeds the 2-filing cap" in str(exc.value)
    assert not (tmp_path / "data" / "edgar_item408" / "unit" / "docs").exists(), \
        "no document directory may exist when the cap stops the run"
