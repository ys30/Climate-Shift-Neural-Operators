"""Frozen E1 v5.13 cross-cohort comparison guards."""
import copy
import pytest
from scripts.compare_cmip6_v513_2040_replication import compare

def make_reports():
    scored={m:{str(w):{
        "weighted_rmse_K":{"mean":1.8},
        "local_p95_recall":{"mean":.6},
        "local_p95_precision":{"mean":.7}}
        for w in (0,1,3)} for m in ("fno","unet")}
    old={"n_pairs":72,"persistence":{"weighted":{"rmse":2.0},
         "local_p95":{"hot_recall":.8},"monthly_hot":{"hot_recall":.8}},
         "models":scored}
    awi={"status":"first_frozen_historical_to_sparse_SSP_forecast_evaluation",
         "future_years":[2030,2050,2080],"future_month_days":[8,20],
         "ssp_used_for_fitting_or_selection":False,
         "checkpoint_sha256":{str(i):"sha" for i in range(18)},
         "scenarios":{s:{"groups":{"all":copy.deepcopy(old)}} for s in ("ssp245","ssp585")}}
    bcc={"status":"first_independent_GCM_BCC_Historical_to_sparse_SSP_frozen_forecast",
         "future_years":[2030,2050,2080],"future_month_days":[8,20],
         "ssp_used_for_fitting_or_checkpoint_selection":False,
         "checkpoint_sha":{str(i):"sha" for i in range(18)},
         "scenario_results":{s:{"scores":{"all":copy.deepcopy(old)}} for s in ("ssp245","ssp585")}}
    new={"status":"v512_frozen_historical_neural_forecasts_new_2040_cohort",
         "frozen_training":True,"checkpoint_selection_using_2040":False,
         "date_plan":[3,7,11,15,19,23,27],"year":2040,"rows":[]}
    for g in ("AWI","BCC"):
        for s in ("ssp245","ssp585"):
            models={"persistence":{"rmse_K":2.0,"p95_recall":.8}}
            for m in ("fno","unet"):
                for w in (0,1,3):
                    models[f"{m}_weight{w}"]={
                        k:{"individual":[v,v,v],"mean":v}
                        for k,v in (("rmse_K",1.7),("p95_recall",.65),("p95_precision",.75))}
            new["rows"].append(dict(gcm=g,ssp=s,year=2040,n_pairs=84,models=models))
    return awi,bcc,new

def test_exact_24_new_original_comparisons():
    result=compare(*make_reports())
    assert len(result["rows"])==24
    assert len(result["four_gcm_scenario_summaries"])==4
    assert result["descriptive_counts"]["fno1_skill_positive_2040"]==4
    assert result["rows"][0]["new_relative_skill_pct"]==pytest.approx(15)

def test_incorrect_new_day_rejected():
    a,b,n=make_reports();n["date_plan"]=[8,20]
    with pytest.raises(ValueError,match="verification gate"):compare(a,b,n)

def test_missing_new_scenario_rejected():
    a,b,n=make_reports();n["rows"].pop()
    with pytest.raises(ValueError,match="Missing new"):compare(a,b,n)

def test_duplicate_new_scenario_rejected():
    a,b,n=make_reports();n["rows"].append(copy.deepcopy(n["rows"][0]))
    with pytest.raises(ValueError,match="Duplicate"):compare(a,b,n)

def test_missing_seed_rejected():
    a,b,n=make_reports()
    n["rows"][0]["models"]["fno_weight0"]["rmse_K"]["individual"].pop()
    with pytest.raises(ValueError,match="three frozen"):compare(a,b,n)

def test_change_original_model_inventory_rejected():
    a,b,n=make_reports();a["checkpoint_sha256"].pop("0")
    with pytest.raises(ValueError,match="Unfrozen"):compare(a,b,n)
