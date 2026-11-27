"""E1 v5.22 regression tests for complete native-source summer metrics."""
import copy
import pytest
from scripts.audit_cmip6_v522_cams_contiguous_events import inspect,export

def make():
    scenarios={}
    for s in ("ssp245","ssp585"):
        baseline={"rmse_K":2.0,"mae_K":1.4,"bias_K":0.1,
                  "daily_p95_recall":.85,"daily_p95_precision":.65,
                  "event_day_recall":.8,"event_day_precision":.7}
        models={}
        for m in ("fno","unet"):
            models[m]={}
            for w in ("0","1","3"):
                z={"rmse_K":1.6+int(w)*.01,"mae_K":1.1,
                   "bias_K":.02,"daily_p95_recall":.6,
                   "daily_p95_precision":.7,
                   "event_day_recall":.45+int(w)*.03,
                   "event_day_precision":.72}
                models[m][w]={k:{"individual":[v]*3,"mean":v,"sd_seeds":0.0}
                               for k,v in z.items()}
        scenarios[s]={"n_pairs":91,"source":{
            "n_source_days":91,"n_bounded_regional_native_days_read":92},
            "persistence":baseline,"models":models}
    return {"status":"v520_CAMS_frozen_contiguous_2050_summer_daily_mean_heat_event_pilot",
            "source_model":"CAMS-CSM1-0","year":2050,"source_count_per_ssp":91,
            "ssp_fitted":False,"checkpoint_sha":{str(i):"sha" for i in range(18)},
            "scenarios":scenarios}

def test_all_complete_frozen_model_arms():
    r=inspect(make())
    assert len(r["scenario_findings"])==2
    assert len(r["arm_summaries"])==12
    assert len(r["seed_metric_rows"])==2*2*3*3*7
    assert r["n_independent_GCMs"]==1
    assert all(x["persistence_minus_fno_standard_event_recall_pp"]==pytest.approx(35.) for x in r["scenario_findings"])

def test_reject_missing_ssp():
    d=make();d["scenarios"].pop("ssp585")
    with pytest.raises(ValueError,match="complete frozen"):inspect(d)

def test_reject_future_fitting():
    d=make();d["ssp_fitted"]=True
    with pytest.raises(ValueError,match="complete frozen"):inspect(d)

def test_reject_missing_seed():
    d=make();d["scenarios"]["ssp245"]["models"]["fno"]["0"]["rmse_K"]["individual"].pop()
    with pytest.raises(ValueError,match="three original"):inspect(d)

def test_reject_bogus_seed_mean():
    d=make();d["scenarios"]["ssp245"]["models"]["fno"]["0"]["event_day_recall"]["mean"]=.99
    with pytest.raises(ValueError,match="mean changed"):inspect(d)

def test_reject_probability_over_one():
    d=make();d["scenarios"]["ssp585"]["persistence"]["event_day_recall"]=1.1
    with pytest.raises(ValueError,match="outside"):inspect(d)

def test_save_three_tables_and_draft(tmp_path):
    r=inspect(make());out=tmp_path/"output"
    export(r,out)
    assert len(list(out.glob("*.csv")))==3
    txt=(out/"v522_CAMS_contiguous_event_results.md").read_text()
    assert "one CAMS member" in txt
    assert "Precision" in txt
