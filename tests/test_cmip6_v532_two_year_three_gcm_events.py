"""E1 v5.32 cross-year audit: independent GCM units and locked checkpoints."""
import json
import pytest
from scripts.audit_cmip6_v532_two_year_three_gcm_events import analyze,write,Y2040
from tests.test_cmip6_v525_three_gcm_contiguous_events import fixture

def reports():
    old=fixture()
    data={}
    for g,original in zip(("AWI","BCC","CAMS"),old):
        from copy import deepcopy
        older=deepcopy(original)
        older["year"]=2040
        older["status"]=Y2040[g][0]
        for s in ("ssp245","ssp585"):
            older["scenarios"][s]["source"]["first_target"]="2040-06-02"
            older["scenarios"][s]["source"]["last_target"]="2040-08-31"
        data[g]={2040:older,2050:original}
    return data

def test_complete_72_arm_two_year_comparison():
    r=analyze(reports())
    assert r["year_GCM_SSP_cells"]==12
    assert len(r["complete_neural_arms"])==72
    assert len(r["paired_year_findings"])==6
    assert r["original_frozen_neural_checkpoint_count"]==54
    assert r["independent_GCM_count"]==3

def test_original_frozen_sha_must_match_years():
    d=reports()
    d["AWI"][2040]["checkpoint_sha"]["0"]="tamper"
    with pytest.raises(ValueError,match="hashes differ"):analyze(d)

def test_missing_year_refused():
    d=reports()
    del d["CAMS"][2040]
    with pytest.raises(ValueError,match="paired"):analyze(d)

def test_no_false_independence_from_12_scenario_cells():
    r=analyze(reports())
    assert r["independent_GCM_count"]==3
    assert len(r["scenario_findings"])==12
    assert set(r["GCM_level_consistency_counts"])==set(r["cell_counts"])

def test_export_all_72_and_12_rows(tmp_path):
    d=analyze(reports());out=tmp_path/"v532"
    write(d,out)
    import csv
    with (out/"v532_all_72_model_scenario_year_rows.csv").open() as f:
        assert len(list(csv.DictReader(f)))==72
    with (out/"v532_12_scenario_year_findings.csv").open() as f:
        assert len(list(csv.DictReader(f)))==12
    assert "only three independent GCM" in (out/"v532_two_year_three_GCM_results.md").read_text()
    with pytest.raises(FileExistsError):write(d,out)
