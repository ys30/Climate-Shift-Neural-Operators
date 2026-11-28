"""E1 v5.38 paired 2->5-day changes, not only sign-count consistency."""
from copy import deepcopy
import csv
import pytest
from scripts.render_cmip6_v538_duration_manuscript import validate,render
from scripts.audit_cmip6_v537_three_gcm_duration_sensitivity import audit
from tests.test_cmip6_v537_three_gcm_duration_sensitivity import sample

def fixture():
    return audit(sample())

def test_full_three_gcm_twelve_pair_contract():
    rows,pairs=validate(fixture())
    assert len(rows)==36 and len(pairs)==12

def test_reject_tampered_archive_paired_changes():
    r=fixture()
    r["paired_two_vs_five_day_changes"][0]["five_minus_two_day_gap_change_pp"]=5.
    with pytest.raises(ValueError,match="paired duration effect changed"):validate(r)

def test_reject_missing_original_gcm_ssp_year():
    r=fixture();r["scenario_duration_findings"].pop()
    with pytest.raises(ValueError,match="36 duration cases"):validate(r)

def test_render_counterdirection_and_full_report(tmp_path):
    pytest.importorskip("matplotlib")
    r=fixture()
    # Create a single 5-day smaller persistence-FNO Recall deficit and propagate source 5-day gap.
    row=next(x for x in r["scenario_duration_findings"] if (x["gcm"],x["year"],x["scenario"],x["min_days"])==("AWI",2040,"ssp245",5))
    row["fno_1x_event_recall"]+=.02
    row["persistence_minus_fno_1x_recall_pp"]-=2.
    row["fno_4x_minus_1x_recall_pp"]-=2.
    pair=next(x for x in r["paired_two_vs_five_day_changes"] if (x["gcm"],x["year"],x["scenario"])==("AWI",2040,"ssp245"))
    pair["recall_gap_5day_pp"]-=2.
    pair["five_minus_two_day_gap_change_pp"]-=2.
    pair["fno_4x_gain_5day_pp"]-=2.
    pair["five_minus_two_day_hotweight_gain_change_pp"]-=2.
    rows,pairs=validate(r)
    out=tmp_path/"paper"
    summary=render(rows,pairs,out)
    assert summary["persistence_minus_FNO_recall_gap_5d_minus_2d"]["negative"]==1
    assert len(list(out.glob("figure_v538_*.png")))==3
    with (out/"table_v538_12_paired_duration_changes.csv").open() as f:
        assert len(list(csv.DictReader(f)))==12
    assert "NOT 12 or 36" in (out/"E1_v538_three_GCM_duration_manuscript_results.md").read_text()
    with pytest.raises(FileExistsError):render(rows,pairs,out)
