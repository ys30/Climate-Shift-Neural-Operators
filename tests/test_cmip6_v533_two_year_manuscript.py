"""E1 v5.33 real-source plot regressions; optional plotting dependency."""
import csv
import pytest
from scripts.render_cmip6_v533_two_year_manuscript import verify,make_plots
from scripts.audit_cmip6_v532_two_year_three_gcm_events import analyze
from tests.test_cmip6_v532_two_year_three_gcm_events import reports

def fixture():
    return analyze(reports())

def test_all_12_paired_year_cases():
    x=verify(fixture())
    assert len(x)==12
    assert len({(r["gcm"],r["year"],r["scenario"]) for r in x})==12

def test_source_checkpoint_count_is_frozen():
    r=fixture();r["original_frozen_neural_checkpoint_count"]=53
    with pytest.raises(ValueError,match="complete frozen"):verify(r)

def test_reject_modified_original_recall_comparison():
    r=fixture()
    r["scenario_findings"][0]["persistence_minus_fno_recall_pp"]+=3
    with pytest.raises(ValueError,match="gap corrupted"):verify(r)

def test_counterexample_not_suppressed_and_images(tmp_path):
    pytest.importorskip("matplotlib")
    d=fixture()
    d["scenario_findings"][0]["four_x_minus_one_x_precision_pp"]=.15
    d["scenario_findings"][1]["four_x_minus_one_x_precision_pp"]=-2
    rows=verify(d)
    out=tmp_path/"paper"
    exc=make_plots(rows,out)
    assert any(r[3]==.15 for r in exc)
    assert len(list(out.glob("figure_v533_*.png")))==2
    with (out/"table_v533_12_two_year_event_cases.csv").open() as f:
        assert len(list(csv.DictReader(f)))==12
    assert "Precision counterexample" in (out/"E1_v533_two_year_three_GCM_results.md").read_text()
    with pytest.raises(FileExistsError):make_plots(rows,out)
