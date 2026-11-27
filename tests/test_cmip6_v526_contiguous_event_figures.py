"""E1 v5.26 source-gated paper visualization tests."""
import copy,csv
import pytest
from scripts.render_cmip6_v526_contiguous_event_figures import verify,render
from tests.test_cmip6_v525_three_gcm_contiguous_events import fixture
from scripts.audit_cmip6_v525_three_gcm_contiguous_events import compare

def report():
    return compare(*fixture())

def test_original_report_has_all_six_GCM_scenarios():
    rows=verify(report())
    assert len(rows)==6
    assert {r["gcm"] for r in rows}=={"AWI","BCC","CAMS"}

def test_corrupted_scenario_rejected():
    d=report()
    d["scenario_findings"][0]["gcm"]="CAMS"
    with pytest.raises(ValueError,match="Missing independent"):verify(d)

def test_corrupted_recall_gap_rejected():
    d=report()
    d["scenario_findings"][0]["persistence_minus_standard_fno_event_recall_pp"]=100.
    with pytest.raises(ValueError,match="Stored event gap"):verify(d)

def test_refit_prohibited():
    d=report();d["ssp_used_for_fitting_or_selection"]=True
    with pytest.raises(ValueError,match="complete v5.25"):verify(d)

def test_generate_png_csv_and_results(tmp_path):
    # matplotlib is an optional manuscript-figure dependency. The dedicated
    # v5.26 workflow installs it and executes this test without skipping.
    pytest.importorskip("matplotlib")
    out=tmp_path/"manuscript"
    render(verify(report()),out)
    assert len(list(out.glob("*.png")))==3
    assert all(p.stat().st_size>5000 for p in out.glob("*.png"))
    with (out/"table_v526_three_gcm_2050_continuous_events.csv").open() as f:
        assert len(list(csv.DictReader(f)))==6
    text=(out/"E1_v526_manuscript_continuous_event_results.md").read_text()
    assert "not six independent GCM replicates" in text
    with pytest.raises(FileExistsError):render(verify(report()),out)
