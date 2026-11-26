"""v5.19 tests source fidelity, all three independent GCMs, seeded score gates."""
import copy
import pytest
from scripts.audit_cmip6_v519_three_gcm_ssp import compare,cams_group
from tests.test_cmip6_v513_2040_replication import make_reports

def fixtures():
    awi,bcc,_=make_reports()
    # preserve original two GCM schemas and add a genuine native CAMS-style SSP score
    scenarios={}
    for scenario in ("ssp245","ssp585"):
        scored={m:{str(w):{
            "weighted_rmse_K":{"mean":2.0},
            "local_p95_recall":{"mean":.6},
            "local_p95_precision":{"mean":.7}}
            for w in (0,1,3)} for m in ("fno","unet")}
        scenarios[scenario]={"scores":{"all":{
           "n_pairs":72,
           "persistence":{"weighted":{"rmse":2.5},"monthly_hot":{"hot_recall":.8}},
           "models":scored}}}
    cams={
      "status":"v518_CAMS_Historical_frozen_sparse_SSP_predictive_skill",
      "source_model":"CAMS-CSM1-0",
      "ssp_used_for_fitting_or_checkpoint_selection":False,
      "checkpoint_sha":{str(i):"sha" for i in range(18)},
      "future_years":[2030,2050,2080],
      "future_month_days":[8,20],
      "scenario_results":scenarios}
    return awi,bcc,cams

def test_three_GCM_six_SSP_36_frozen_arms():
    result=compare(*fixtures())
    assert result["independent_GCM_count"]==3
    assert len(result["arms"])==36
    assert len(result["gcm_ssp_findings"])==6
    assert len(result["gcm_grouped_descriptive_findings"])==3
    assert set(r["gcm"] for r in result["arms"])=={"AWI","BCC","CAMS"}
    assert result["descriptive_counts"]["gcm_with_both_ssp_fno1_rmse_advantage"]==3

def test_missing_CAMS_checkpoint_rejected():
    a,b,c=fixtures();c["checkpoint_sha"].pop("1")
    with pytest.raises(ValueError,match="Unverified"):compare(a,b,c)

def test_missing_CAMS_SSP_rejected():
    a,b,c=fixtures();c["scenario_results"].pop("ssp585")
    with pytest.raises(ValueError,match="Incomplete CAMS"):compare(a,b,c)

def test_future_training_prohibited():
    a,b,c=fixtures();c["ssp_used_for_fitting_or_checkpoint_selection"]=True
    with pytest.raises(ValueError,match="Unverified"):compare(a,b,c)

def test_nonfinite_future_scores_rejected():
    a,b,c=fixtures()
    c["scenario_results"]["ssp585"]["scores"]["all"]["models"]["fno"]["0"]["weighted_rmse_K"]["mean"]=float("nan")
    with pytest.raises(ValueError,match="Non-finite"):compare(a,b,c)

def test_source_specific_baseline_is_never_pooled():
    a,b,c=fixtures()
    c["scenario_results"]["ssp245"]["scores"]["all"]["persistence"]["weighted"]["rmse"]=5.0
    d=compare(a,b,c)
    r=next(x for x in d["arms"] if x["gcm"]=="CAMS" and x["ssp"]=="ssp245")
    assert r["relative_rmse_skill_pct"]==pytest.approx(60.)
