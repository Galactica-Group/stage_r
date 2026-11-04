"""
================================================================================
Script:       test_edgar_item408_fsn.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Fixture-only battery on the v1.1.0 bulk-dataset route of
              tools/edgar_item408_probe.py: the Item 408 fact filter, dimension
              parsing and resolution, the per-(adsh, individual) pivot into the
              R5-0 row schema, the empty-quarter case, the download manifest,
              and the contextRef-over-splitting correction on the HTML path.
Entry Point:  No — pytest
Dependencies: stdlib, pytest, pandas; the probe module via importlib
Last Sprint:  Sprint R5-1 — bulk dataset route (2026-11-04)
================================================================================

No network. The bulk archive is a synthetic zip built in the test, shaped from
the real 2024q1 members, with hand-checkable answers.

The mutations this battery is built to catch are: a fact filter that accepts a
filer's custom same-named tag or drops the numeric member; a pivot that keys on
the dimension hash instead of the Individual member, so one person's duration
facts and instant facts land in two rows; a pivot that merges two different
people because their hashes collide in the test; an empty quarter that produces
a row anyway; a download that overwrites a banked archive; and the HTML
grouping regression where the aggregate-securities amount, tagged in its own
contextRef, splits one arrangement into two rows.
================================================================================
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "edgar_item408_probe.py"


def _module():
    spec = importlib.util.spec_from_file_location("edgar_item408_probe_fsn", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _module()


# ---------------------------------------------------------------------------
# The fact filter
# ---------------------------------------------------------------------------

def test_the_fact_filter_takes_the_item_408_family_in_the_ecd_taxonomy(m):
    for tag in ("TrdArrIndName", "TrdArrIndTitle", "TrdArrAdoptionDate",
                "TrdArrTerminationDate", "TrdArrDuration",
                "TrdArrSecuritiesAggAvailAmt", "TrdArrExpirationDate",
                "Rule10b51ArrAdoptedFlag", "Rule10b51ArrTrmntdFlag",
                "NonRule10b51ArrAdoptedFlag", "NonRule10b51ArrTrmntdFlag"):
        assert m.is_item408_fact(tag, "ecd/2023") is True, tag


def test_the_fact_filter_rejects_a_custom_tag_of_the_same_name(m):
    """A filer-defined tag in its own namespace is not the SEC concept."""
    assert m.is_item408_fact("TrdArrIndName", "enph/2023") is False
    assert m.is_item408_fact("Rule10b51ArrTrmntdFlag", "0001463101-24-000024") is False


def test_the_fact_filter_rejects_other_ecd_concepts_and_other_taxonomies(m):
    # The pay-versus-performance concepts are ecd but not Item 408.
    assert m.is_item408_fact("PeerGroupTotalShareholderRtnAmt", "ecd/2023") is False
    assert m.is_item408_fact("CoSelectedMeasureAmt", "ecd/2023") is False
    assert m.is_item408_fact("Assets", "us-gaap/2023") is False
    assert m.is_item408_fact("AmendmentFlag", "dei/2023") is False
    assert m.is_item408_fact(None, None) is False
    assert m.is_item408_fact("TrdArrIndName", None) is False


# ---------------------------------------------------------------------------
# Dimension parsing and resolution
# ---------------------------------------------------------------------------

def test_dimension_segments_parse_into_axis_member_pairs(m):
    assert m.parse_dim_segments("Individual=DavidRanhoff;") == {
        "Individual": "DavidRanhoff"
    }
    assert m.parse_dim_segments("") == {}
    assert m.parse_dim_segments(None) == {}
    multi = m.parse_dim_segments("Individual=JaneRoe;TrdArrByIndividual=Plan1;")
    assert multi == {"Individual": "JaneRoe", "TrdArrByIndividual": "Plan1"}
    # A malformed fragment is skipped, not guessed at.
    assert m.parse_dim_segments("garbage;Individual=JaneRoe;") == {
        "Individual": "JaneRoe"
    }


def test_the_row_key_is_the_canonical_dimension_set_and_falls_back_to_the_hash(m):
    assert m.fsn_individual_key("0xabc", "Individual=DavidRanhoff;") == \
        "Individual=DavidRanhoff"
    assert m.fsn_individual_key("0xabc", "IndividualAxis=JaneRoe;") == \
        "Individual=JaneRoe"
    # Entity-level: no dimensions at all, so the hash keeps the row separate.
    assert m.fsn_individual_key("0x00000000", "") == "0x00000000"
    # Another axis is kept, not discarded and not mistaken for the individual.
    assert m.fsn_individual_key("0xdef", "AwardType=RSU;") == "AwardType=RSU"


def test_the_canonical_key_is_the_same_from_either_source(m):
    """The HTML and bulk spellings of one dimension set must agree exactly."""
    fsn = m.canonical_dim_key(m.parse_dim_segments(
        "Individual=SanjayMehrotra;TradingArr=January302026ModifiedPlan;"))
    html = m.canonical_dim_key({"IndividualAxis": "SanjayMehrotraMember",
                                "TradingArrAxis": "January302026ModifiedPlanMember"})
    assert fsn == html == "Individual=SanjayMehrotra;TradingArr=January302026ModifiedPlan"
    # Axis order in the source must not change the key.
    assert m.canonical_dim_key({"TradingArrAxis": "P1", "IndividualAxis": "X"}) == \
        m.canonical_dim_key({"IndividualAxis": "X", "TradingArrAxis": "P1"})
    assert m.canonical_dim_key({}) == ""
    assert m.parse_canonical_dim_key("Individual=X;TradingArr=P1") == \
        {"Individual": "X", "TradingArr": "P1"}


def test_the_dimension_resolution_separates_two_individuals(m):
    """Two people in one filing must key to two different rows."""
    a = m.fsn_individual_key("0x111", "Individual=JaneRoe;")
    b = m.fsn_individual_key("0x222", "Individual=JohnDoe;")
    assert a != b and (a, b) == ("Individual=JaneRoe", "Individual=JohnDoe")
    # The same person tagged under two hashes keys to ONE row.
    assert m.fsn_individual_key("0x333", "Individual=JaneRoe;") == a
    # But two ARRANGEMENTS of one person stay apart - the R5-1 correction.
    p1 = m.fsn_individual_key("0x444", "Individual=JaneRoe;TradingArr=PlanA;")
    p2 = m.fsn_individual_key("0x555", "Individual=JaneRoe;TradingArr=PlanB;")
    assert p1 != p2
    assert p1 != a and p2 != a


# ---------------------------------------------------------------------------
# A synthetic bulk archive
# ---------------------------------------------------------------------------

TXT_HEADER = ["adsh", "tag", "version", "ddate", "qtrs", "iprx", "lang", "dcml",
              "durp", "datp", "dimh", "dimn", "coreg", "escaped", "srclen",
              "txtlen", "footnote", "footlen", "context", "value"]
NUM_HEADER = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "dimh", "iprx",
              "value", "footnote", "footlen", "dimn", "coreg", "durp", "datp", "dcml"]
SUB_HEADER = ["adsh", "cik", "name", "form", "period", "filed", "detail"]
DIM_HEADER = ["dimhash", "segments", "segt"]

ADSH = "0001463101-24-000024"


def _tsv(header, rows):
    out = ["\t".join(header)]
    for r in rows:
        out.append("\t".join(str(r.get(c, "")) for c in header))
    return "\n".join(out) + "\n"


def _txt(tag, dimh, value, version="ecd/2023", adsh=ADSH):
    return {"adsh": adsh, "tag": tag, "version": version, "ddate": "20231231",
            "dimh": dimh, "value": value}


def _build_archive(path, txt_rows, num_rows, sub_rows, dim_rows):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("txt.tsv", _tsv(TXT_HEADER, txt_rows))
        z.writestr("num.tsv", _tsv(NUM_HEADER, num_rows))
        z.writestr("sub.tsv", _tsv(SUB_HEADER, sub_rows))
        z.writestr("dim.tsv", _tsv(DIM_HEADER, dim_rows))
    return path


@pytest.fixture
def two_individual_archive(tmp_path):
    """One filing, two individuals, one entity-level flag row.

    JaneRoe's facts are split across two dimension hashes - a duration hash and
    an instant hash, exactly as a real filer tags them - so the pivot must merge
    them into one row.
    """
    txt = [
        _txt("Rule10b51ArrAdoptedFlag", "0x00000000", "false"),
        _txt("Rule10b51ArrTrmntdFlag", "0x00000000", "true"),
        _txt("TrdArrIndName", "0xJANE_D", "Jane Roe"),
        _txt("TrdArrIndTitle", "0xJANE_D", "Chief Financial Officer"),
        _txt("TrdArrTerminationDate", "0xJANE_D", "May 13, 2024"),
        _txt("TrdArrDuration", "0xJANE_D", "P180D"),
        _txt("MtrlTermsOfTrdArrTextBlock", "0xJANE_D",
             "<p>On May 13, 2024 Jane Roe terminated a plan.</p>"),
        _txt("TrdArrIndName", "0xJOHN_D", "John Doe"),
        _txt("TrdArrIndTitle", "0xJOHN_D", "Director"),
        _txt("TrdArrAdoptionDate", "0xJOHN_D", "March 4, 2024"),
        # A decoy: right tag name, filer's own namespace.
        _txt("TrdArrIndName", "0xJOHN_D", "Not The SEC Concept", version="enph/2023"),
        # A decoy: ecd taxonomy but not Item 408.
        _txt("CoSelectedMeasureAmt", "0x00000000", "123"),
    ]
    num = [
        # Jane's amount, tagged under her INSTANT hash, not her duration hash.
        {"adsh": ADSH, "tag": "TrdArrSecuritiesAggAvailAmt", "version": "ecd/2023",
         "ddate": "20231231", "dimh": "0xJANE_I", "value": "40000.0000"},
        {"adsh": ADSH, "tag": "Assets", "version": "us-gaap/2023",
         "ddate": "20231231", "dimh": "0x00000000", "value": "999"},
    ]
    sub = [{"adsh": ADSH, "cik": "1463101", "name": "ENPHASE ENERGY, INC.",
            "form": "10-K", "period": "20231231", "filed": "20240209", "detail": "1"}]
    dim = [
        {"dimhash": "0x00000000", "segments": "", "segt": "0"},
        {"dimhash": "0xJANE_D", "segments": "Individual=JaneRoe;", "segt": "1"},
        {"dimhash": "0xJANE_I", "segments": "Individual=JaneRoe;", "segt": "1"},
        {"dimhash": "0xJOHN_D", "segments": "Individual=JohnDoe;", "segt": "1"},
    ]
    return _build_archive(tmp_path / "2024q1_notes.zip", txt, num, sub, dim)


def test_reading_the_archive_keeps_only_item_408_ecd_facts(m, two_individual_archive):
    facts, subs, dims = m.read_fsn_item408(str(two_individual_archive))
    tags = sorted({f["tag"] for f in facts})
    assert "CoSelectedMeasureAmt" not in tags
    assert "Assets" not in tags
    assert "TrdArrSecuritiesAggAvailAmt" in tags, "the numeric member must be read"
    # The custom-namespace decoy is excluded, so TrdArrIndName appears twice only.
    assert sum(1 for f in facts if f["tag"] == "TrdArrIndName") == 2
    assert all(f["version"].startswith("ecd") for f in facts)
    assert set(subs) == {ADSH}
    # Only the hashes an Item 408 fact actually uses are resolved.
    assert set(dims) == {"0x00000000", "0xJANE_D", "0xJANE_I", "0xJOHN_D"}


def test_the_pivot_gives_one_row_per_individual_plus_the_entity_row(
        m, two_individual_archive):
    facts, subs, dims = m.read_fsn_item408(str(two_individual_archive))
    rows = m.pivot_fsn_rows(facts, subs, dims)
    keys = sorted(r["context_ref"] for r in rows)
    assert keys == ["0x00000000", "Individual=JaneRoe", "Individual=JohnDoe"], \
        "Jane's two hashes must merge into one row"
    assert len(rows) == 3


def test_the_pivot_fills_the_r5_0_row_schema(m, two_individual_archive):
    facts, subs, dims = m.read_fsn_item408(str(two_individual_archive))
    rows = {r["context_ref"]: r for r in m.pivot_fsn_rows(facts, subs, dims)}
    assert set(rows["Individual=JaneRoe"]) == set(m.EVENT_COLUMNS)

    jane = rows["Individual=JaneRoe"]
    assert jane["adsh"] == ADSH
    assert jane["cik"] == "0001463101", "the sub.tsv CIK is zero-padded to ten"
    assert jane["form"] == "10-K"
    assert jane["file_date"] == "2024-02-09"
    assert jane["period_ending"] == "2023-12-31"
    assert jane["individual_name"] == "Jane Roe"
    assert jane["individual_title"] == "Chief Financial Officer"
    assert jane["termination_date"] == "2024-05-13"
    assert jane["termination_date_raw"] == "May 13, 2024"
    assert jane["duration_text"] == "P180D"
    # The instant-context amount reached her row.
    assert jane["securities_agg_available"] == "40000.0000"
    # The text block is stored with its markup stripped, and classified not at all.
    assert jane["raw_text"] == "On May 13, 2024 Jane Roe terminated a plan."

    john = rows["Individual=JohnDoe"]
    assert john["individual_name"] == "John Doe"
    assert john["adoption_date"] == "2024-03-04"
    assert john["termination_date"] is None
    assert john["securities_agg_available"] is None

    entity = rows["0x00000000"]
    assert entity["individual_name"] is None
    assert entity["rule10b51_adopted_flag"] is False
    assert entity["rule10b51_terminated_flag"] is True


def test_an_archive_with_no_ecd_facts_yields_zero_rows(m, tmp_path):
    txt = [_txt("AmendmentFlag", "0x00000000", "false", version="dei/2023")]
    num = [{"adsh": ADSH, "tag": "Assets", "version": "us-gaap/2023",
            "ddate": "20231231", "dimh": "0x00000000", "value": "1"}]
    sub = [{"adsh": ADSH, "cik": "1463101", "name": "X", "form": "10-K",
            "period": "20231231", "filed": "20240209", "detail": "1"}]
    dim = [{"dimhash": "0x00000000", "segments": "", "segt": "0"}]
    p = _build_archive(tmp_path / "2009q1_notes.zip", txt, num, sub, dim)
    facts, subs, dims = m.read_fsn_item408(str(p))
    assert facts == []
    assert subs == {} and dims == {}
    assert m.pivot_fsn_rows(facts, subs, dims) == []


def test_a_missing_member_does_not_raise(m, tmp_path):
    p = tmp_path / "bare_notes.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("readme.htm", "<html></html>")
    assert m.read_fsn_item408(str(p)) == ([], {}, {})


# ---------------------------------------------------------------------------
# The download manifest
# ---------------------------------------------------------------------------

def test_the_download_manifest_round_trips_and_skips_a_banked_archive(
        m, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(m, "REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("EDGAR_CONTACT", "Acme Research ops@acme.test")
    name = "2024q1_notes.zip"
    fsn = pathlib.Path(m.fsn_dir())
    fsn.mkdir(parents=True, exist_ok=True)
    (fsn / name).write_bytes(b"already here")
    m.write_manifest(str(fsn / "archives.json"),
                     {"archives": {name: {"bytes": len(b"already here"),
                                          "sha256": "deadbeef", "status": 200}}})

    import argparse
    args = argparse.Namespace(vintage="unit", archives=[name], download=True,
                              build=False, rate=6)

    class Boom:
        def get(self, *a, **k):
            raise AssertionError("a banked archive must not be re-downloaded")

    monkeypatch.setitem(__import__("sys").modules, "requests", Boom())
    rc = m.cmd_fsn(args)
    assert rc == 0
    assert "skip %s" % name in capsys.readouterr().out
    assert (fsn / name).read_bytes() == b"already here"
    back = json.loads((fsn / "archives.json").read_text())
    assert back["archives"][name]["sha256"] == "deadbeef"


def test_the_archive_url_is_built_under_the_sec_dataset_path(m):
    assert m.fsn_url("2024q1_notes.zip") == (
        "https://www.sec.gov/files/dera/data/financial-statement-notes-data-sets/"
        "2024q1_notes.zip"
    )
    assert m.fsn_url("2026_09_notes.zip").startswith("https://www.sec.gov/")


# ---------------------------------------------------------------------------
# The HTML grouping correction (R5-0 keyed on contextRef and over-split)
# ---------------------------------------------------------------------------

SPLIT_CONTEXTS = """
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>
<div style="display:none">
<xbrli:context id="c-493"><xbrli:entity>
 <xbrli:identifier scheme="http://www.sec.gov/CIK">0001463101</xbrli:identifier>
 <xbrli:segment><xbrldi:explicitMember dimension="ecd:IndividualAxis">enph:DavidRanhoffMember</xbrldi:explicitMember></xbrli:segment>
</xbrli:entity><xbrli:period><xbrli:startDate>2023-10-01</xbrli:startDate>
<xbrli:endDate>2023-12-31</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:context id="c-494"><xbrli:entity>
 <xbrli:identifier scheme="http://www.sec.gov/CIK">0001463101</xbrli:identifier>
 <xbrli:segment><xbrldi:explicitMember dimension="ecd:IndividualAxis">enph:DavidRanhoffMember</xbrldi:explicitMember></xbrli:segment>
</xbrli:entity><xbrli:period><xbrli:instant>2023-12-31</xbrli:instant></xbrli:period></xbrli:context>
<xbrli:context id="c-1"><xbrli:entity>
 <xbrli:identifier scheme="http://www.sec.gov/CIK">0001463101</xbrli:identifier>
</xbrli:entity><xbrli:period><xbrli:instant>2023-12-31</xbrli:instant></xbrli:period></xbrli:context>
</div>
<table>
 <tr><td><ix:nonNumeric contextRef="c-493" name="ecd:TrdArrIndName">David Ranhoff</ix:nonNumeric></td>
     <td><ix:nonNumeric contextRef="c-493" name="ecd:TrdArrAdoptionDate" format="ixt:date-monthname-day-year-en">December 4, 2023</ix:nonNumeric></td>
     <td><ix:nonFraction contextRef="c-494" name="ecd:TrdArrSecuritiesAggAvailAmt" decimals="0">40,000</ix:nonFraction></td></tr>
</table>
<p><ix:nonNumeric contextRef="c-1" name="ecd:Rule10b51ArrTrmntdFlag" format="ixt:fixed-false">false</ix:nonNumeric></p>
</body></html>
"""


def test_context_dimensions_are_read_from_the_document(m):
    cd = m.extract_context_dimensions(SPLIT_CONTEXTS)
    assert cd["c-493"] == {"IndividualAxis": "DavidRanhoffMember"}
    assert cd["c-494"] == {"IndividualAxis": "DavidRanhoffMember"}
    assert cd["c-1"] == {}


def test_two_contexts_for_one_person_collapse_into_one_row(m):
    """The R5-0 defect: the instant-context amount made a second row."""
    meta = {"adsh": ADSH, "cik": "0001463101", "form": "10-K",
            "file_date": "2024-02-09", "period_ending": "2023-12-31"}
    rows = m.event_rows_for_filing(meta, SPLIT_CONTEXTS)
    keys = sorted(r["context_ref"] for r in rows)
    assert keys == ["Individual=DavidRanhoff", "c-1"]
    david = next(r for r in rows if r["context_ref"] == "Individual=DavidRanhoff")
    assert david["individual_name"] == "David Ranhoff"
    assert david["adoption_date"] == "2023-12-04"
    assert david["securities_agg_available"] == "40,000", \
        "the instant-context amount must reach the individual's row"
    entity = next(r for r in rows if r["context_ref"] == "c-1")
    assert entity["rule10b51_terminated_flag"] is False
    assert entity["individual_name"] is None


def test_keying_on_contextref_would_over_split_the_same_document(m):
    """Non-vacuity: the un-resolved grouping is the behaviour being corrected."""
    facts = m.extract_ecd_facts(SPLIT_CONTEXTS)
    unresolved = m.facts_to_rows(facts)
    resolved = m.facts_to_rows(facts, m.extract_context_dimensions(SPLIT_CONTEXTS))
    assert len(unresolved) == 3
    assert len(resolved) == 2


def test_a_document_without_context_blocks_keeps_its_contextref_keys(m):
    """Backwards compatibility: no context section means no regrouping."""
    html = ('<table><tr><td><ix:nonNumeric contextRef="c-71" '
            'name="ecd:TrdArrIndName">Jane Roe</ix:nonNumeric></td>'
            '<td><ix:nonNumeric contextRef="c-72" '
            'name="ecd:TrdArrIndName">John Doe</ix:nonNumeric></td></tr></table>')
    assert m.extract_context_dimensions(html) == {}
    rows = m.facts_to_rows(m.extract_ecd_facts(html),
                           m.extract_context_dimensions(html))
    assert [r["context_ref"] for r in rows] == ["c-71", "c-72"]


def test_the_payload_predicate_adds_only_the_text_block(m):
    """The counting predicate stays exact; the read path widens by one tag."""
    assert m.is_item408_fact("MtrlTermsOfTrdArrTextBlock", "ecd/2023") is False
    assert m.is_item408_payload("MtrlTermsOfTrdArrTextBlock", "ecd/2023") is True
    # Still namespace-gated, and still nothing else.
    assert m.is_item408_payload("MtrlTermsOfTrdArrTextBlock", "enph/2023") is False
    assert m.is_item408_payload("CoSelectedMeasureAmt", "ecd/2023") is False
    assert m.is_item408_payload("Assets", "us-gaap/2023") is False
    # Everything the narrow predicate takes, the wide one takes too.
    for tag in ("TrdArrIndName", "Rule10b51ArrTrmntdFlag", "TrdArrTerminationDate"):
        assert m.is_item408_payload(tag, "ecd/2023") is True


# ---------------------------------------------------------------------------
# Two arrangements for one person, and individual-level inheritance
# ---------------------------------------------------------------------------

@pytest.fixture
def two_arrangement_archive(tmp_path):
    """One person, two plans, shaped from the real Micron 10-Q (0000723125-26-000006).

    Sanjay's November plan was terminated and his January plan adopted, each
    tagged against (Individual, TradingArr). Keying on the individual alone
    collapses them and one date silently wins. Dates are written with the space
    collapsed out, exactly as the bulk dataset ships them.
    """
    txt = [
        _txt("Rule10b51ArrTrmntdFlag", "0x00000000", "true"),
        _txt("TrdArrIndName", "0xNOV", "Sanjay Mehrotra"),
        _txt("TrdArrIndTitle", "0xNOV", "Chairman of the Board"),
        _txt("TrdArrAdoptionDate", "0xNOV", "November8, 2025"),
        _txt("TrdArrTerminationDate", "0xNOV", "January30, 2026"),
        _txt("TrdArrIndName", "0xJAN", "Sanjay Mehrotra"),
        _txt("TrdArrIndTitle", "0xJAN", "Chairman of the Board"),
        _txt("TrdArrAdoptionDate", "0xJAN", "January30, 2026"),
        _txt("TrdArrExpirationDate", "0xJAN", "May1, 2027"),
    ]
    num = [
        {"adsh": ADSH, "tag": "TrdArrSecuritiesAggAvailAmt", "version": "ecd/2023",
         "ddate": "20260228", "dimh": "0xNOV", "value": "200000.0000"},
        {"adsh": ADSH, "tag": "TrdArrSecuritiesAggAvailAmt", "version": "ecd/2023",
         "ddate": "20260228", "dimh": "0xJAN", "value": "150000.0000"},
    ]
    sub = [{"adsh": ADSH, "cik": "723125", "name": "MICRON", "form": "10-Q",
            "period": "20260228", "filed": "20260319", "detail": "1"}]
    dim = [
        {"dimhash": "0x00000000", "segments": "", "segt": "0"},
        {"dimhash": "0xNOV",
         "segments": "Individual=SanjayMehrotra;TradingArr=November82025OriginalPlan;",
         "segt": "0"},
        {"dimhash": "0xJAN",
         "segments": "Individual=SanjayMehrotra;TradingArr=January302026ModifiedPlan;",
         "segt": "0"},
    ]
    return _build_archive(tmp_path / "2026_03_notes.zip", txt, num, sub, dim)


def test_two_arrangements_of_one_person_stay_two_rows(m, two_arrangement_archive):
    facts, subs, dims = m.read_fsn_item408(str(two_arrangement_archive))
    rows = {r["context_ref"]: r for r in m.pivot_fsn_rows(facts, subs, dims)}
    arr = {k: v for k, v in rows.items() if k.startswith("Individual=")}
    assert len(arr) == 2, "one row per arrangement, not one per person"

    nov = rows["Individual=SanjayMehrotra;TradingArr=November82025OriginalPlan"]
    jan = rows["Individual=SanjayMehrotra;TradingArr=January302026ModifiedPlan"]
    assert nov["individual_name"] == jan["individual_name"] == "Sanjay Mehrotra"
    # Each arrangement keeps its own dates and its own amount.
    assert nov["adoption_date"] == "2025-11-08"
    assert nov["termination_date"] == "2026-01-30"
    assert nov["securities_agg_available"] == "200000.0000"
    assert jan["adoption_date"] == "2026-01-30"
    assert jan["termination_date"] is None
    assert jan["expiration_date"] == "2027-05-01"
    assert jan["securities_agg_available"] == "150000.0000"


def test_the_space_collapsed_bulk_dates_parse(m):
    """The bulk dataset ships "December19, 2025"; a required space loses it."""
    assert m.parse_ix_date(None, "December19, 2025") == "2025-12-19"
    assert m.parse_ix_date(None, "March13, 2026") == "2026-03-13"
    assert m.parse_ix_date(None, "May1, 2027") == "2027-05-01"
    assert m.parse_ix_date(None, "November8, 2025") == "2025-11-08"
    # The spaced spelling still works, and nonsense is still rejected.
    assert m.parse_ix_date(None, "May 13, 2024") == "2024-05-13"
    assert m.parse_ix_date(None, "Dember19, 2025") is None


@pytest.fixture
def inheritance_archive(tmp_path):
    """Shaped from the real X4 10-Q (0001501697-25-000043).

    Paula's name, title and termination date are tagged against the Individual
    axis alone, while two amounts are tagged against (Individual, TradingArr).
    """
    txt = [
        _txt("TrdArrIndName", "0xP", "Dr. Paula Ragan"),
        _txt("TrdArrIndTitle", "0xP", "former President and CEO"),
        _txt("TrdArrTerminationDate", "0xP", "August 14, 2025"),
        _txt("Rule10b51ArrTrmntdFlag", "0xP", "true"),
    ]
    num = [
        {"adsh": ADSH, "tag": "TrdArrSecuritiesAggAvailAmt", "version": "ecd/2023",
         "ddate": "20250930", "dimh": "0xP1", "value": "25682.0000"},
        {"adsh": ADSH, "tag": "TrdArrSecuritiesAggAvailAmt", "version": "ecd/2023",
         "ddate": "20250930", "dimh": "0xP2", "value": "313794.0000"},
    ]
    sub = [{"adsh": ADSH, "cik": "1501697", "name": "X4", "form": "10-Q",
            "period": "20250930", "filed": "20251105", "detail": "1"}]
    dim = [
        {"dimhash": "0xP", "segments": "Individual=Dr.PaulaRagan;", "segt": "0"},
        {"dimhash": "0xP1",
         "segments": "Individual=Dr.PaulaRagan;TradingArr=PlanA;", "segt": "0"},
        {"dimhash": "0xP2",
         "segments": "Individual=Dr.PaulaRagan;TradingArr=PlanB;", "segt": "0"},
    ]
    return _build_archive(tmp_path / "2025_11_notes.zip", txt, num, sub, dim)


def test_individual_level_facts_are_inherited_by_each_arrangement(
        m, inheritance_archive):
    facts, subs, dims = m.read_fsn_item408(str(inheritance_archive))
    rows = m.pivot_fsn_rows(facts, subs, dims)
    keys = sorted(r["context_ref"] for r in rows)
    assert keys == ["Individual=Dr.PaulaRagan;TradingArr=PlanA",
                    "Individual=Dr.PaulaRagan;TradingArr=PlanB"], \
        "the individual-only row is redundant once its arrangements exist"
    for r in rows:
        assert r["individual_name"] == "Dr. Paula Ragan"
        assert r["individual_title"] == "former President and CEO"
        assert r["termination_date"] == "2025-08-14"
        assert r["rule10b51_terminated_flag"] is True
    amounts = sorted(r["securities_agg_available"] for r in rows)
    assert amounts == ["25682.0000", "313794.0000"], \
        "each arrangement keeps its own amount; neither is silently dropped"


def test_inheritance_never_overwrites_a_value_the_arrangement_states(m):
    base = {c: None for c in ("individual_name", "termination_date", "adsh",
                              "cik", "form", "file_date", "period_ending",
                              "context_ref")}
    base.update({"context_ref": "Individual=X", "individual_name": "X Person",
                 "termination_date": "2025-01-01"})
    arr = dict(base)
    arr.update({"context_ref": "Individual=X;TradingArr=P1",
                "individual_name": None, "termination_date": "2025-06-30"})
    out = m.apply_individual_inheritance([base, arr])
    assert len(out) == 1
    assert out[0]["individual_name"] == "X Person", "the gap was filled"
    assert out[0]["termination_date"] == "2025-06-30", "the stated value stood"


def test_a_person_with_no_arrangement_row_keeps_their_individual_row(m):
    row = {"context_ref": "Individual=OnlyPerson", "individual_name": "Only Person",
           "adsh": ADSH}
    entity = {"context_ref": "0x00000000", "individual_name": None, "adsh": ADSH}
    out = m.apply_individual_inheritance([row, entity])
    assert len(out) == 2
    assert sorted(r["context_ref"] for r in out) == ["0x00000000",
                                                     "Individual=OnlyPerson"]


def test_keying_on_the_individual_alone_would_lose_an_arrangement(
        m, two_arrangement_archive):
    """Non-vacuity: the collapse this correction prevents, demonstrated."""
    facts, subs, dims = m.read_fsn_item408(str(two_arrangement_archive))
    collapsed = {}
    for f in facts:
        seg = dims.get(f["dimh"], "")
        ind = m.parse_dim_segments(seg).get("Individual") or f["dimh"]
        collapsed.setdefault(ind, []).append(f)
    assert len(collapsed["SanjayMehrotra"]) > 0
    # One bucket for two arrangements: the two adoption dates cannot both survive.
    adopts = {f["value"] for f in collapsed["SanjayMehrotra"]
              if f["tag"] == "TrdArrAdoptionDate"}
    assert adopts == {"November8, 2025", "January30, 2026"}
    assert len([k for k in collapsed if k == "SanjayMehrotra"]) == 1
