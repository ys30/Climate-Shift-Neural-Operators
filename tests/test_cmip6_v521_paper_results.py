"""E1 v5.21: provenance-first figure validation, not inferential pooling."""
import csv
import pytest
from scripts.render_cmip6_v521_paper_results import audited,results_markdown,write_tables

def fixture():
    rows=[]
    for g in ("AWI","BCC","CAMS"):
        for s in ("ssp245","ssp585"):
            rows.append({
              "gcm":g,"ssp":s,"persistence_rmse_K":2.0,
              "fno1_rmse_K":1.8,"fno1_rmse_skill_pct":10.0,
              "fno4_recall_minus_fno1_pp":3.5,
              "fno4_rmse_minus_fno1_K":0.02,
              "persistence_recall_minus_fno4_pp":2.0})
    return {"status":"v519_three_independent_GCM_frozen_sparse_SSP_audit",
        "independent_GCM_count":3,"scenario_cells":6,
        "ssp_used_for_checkpoint_selection":False,
        "gcm_ssp_findings":rows,"arms":[{} for _ in range(36)]}

def test_exact_six_source_scenario_cells():
    rows=audited(fixture())
    assert len(rows)==6
    assert {"AWI","BCC","CAMS"}=={x["gcm"] for x in rows}

def test_no_missing_independent_gcm():
    d=fixture();d["gcm_ssp_findings"][4]["gcm"]="AWI"
    with pytest.raises(ValueError,match="Missing or repeated"):audited(d)

def test_reject_broken_persistence_relative_skill():
    d=fixture();d["gcm_ssp_findings"][0]["fno1_rmse_skill_pct"]=15.
    with pytest.raises(ValueError,match="Broken native"):audited(d)

def test_reject_future_checkpoint_selection():
    d=fixture();d["ssp_used_for_checkpoint_selection"]=True
    with pytest.raises(ValueError,match="Incomplete"):audited(d)

def test_no_six_independent_model_claim():
    draft=results_markdown(audited(fixture()))
    assert "not six independent climate-model replications" in draft
    assert "heatwave persistence" in draft

def test_export_full_v519_rows_with_additional_boolean_field(tmp_path):
    d=fixture()
    for row in d["gcm_ssp_findings"]:
        row["fno1_beats_persistence_rmse"]=True
    path=tmp_path/"manuscript.csv"
    write_tables(audited(d),path)
    with path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==6
    assert "fno1_beats_persistence_rmse" not in rows[0]
    assert {r["gcm"] for r in rows}=={"AWI","BCC","CAMS"}
