"""E1 v5.25 regression tests: three independent original GCM event sources."""
import copy
import csv
import pytest
from scripts.audit_cmip6_v525_three_gcm_contiguous_events import SOURCES,SCENARIOS,compare,export

def fixture():
    reports=[]
    for g in ("AWI","BCC","CAMS"):
        status,model,_,shape=SOURCES[g]
        scenarios={}
        for scenario in SCENARIOS:
            baseline={"rmse_K":2.0,"mae_K":1.2,"bias_K":0.,
                      "daily_p95_recall":.85,"daily_p95_precision":.8,
                      "event_day_recall":.8,"event_day_precision":.82}
            models={}
            for name in ("fno","unet"):
                models[name]={}
                for w in ("0","1","3"):
                    z={"rmse_K":1.8+int(w)*.01,"mae_K":1.0,"bias_K":0.,
                       "daily_p95_recall":.6,"daily_p95_precision":.7,
                       "event_day_recall":.5+int(w)*.05,
                       "event_day_precision":.78}
                    models[name][w]={k:{"individual":[v]*3,"mean":v}
                        for k,v in z.items()}
            meta={"n_source_days":91,"n_bounded_regional_native_days_read":92}
            if shape:meta["native_grid_shape"]=shape
            if g=="AWI":meta["native_calendar"]="proleptic_gregorian"
            if g=="BCC":meta["native_calendar"]="noleap"
            scenarios[scenario]={"n_pairs":91,"source":meta,
                "persistence":baseline,"models":models}
        reports.append({"status":status,"source_model":model,
            "year":2050,"source_count_per_ssp":91,"ssp_fitted":False,
            "checkpoint_sha":{str(i):"fixed" for i in range(18)},
            "scenarios":scenarios})
    return reports

def test_all_36_original_frozen_arms_and_six_scenario_rows():
    r=compare(*fixture())
    assert len(r["neural_arms"])==36
    assert len(r["scenario_findings"])==6
    assert r["frozen_checkpoint_count"]==54
    assert r["GCM_level_counts"]["persistence_event_recall_exceeds_standard_FNO_both_SSP"]==3
    assert r["GCM_level_counts"]["strong_tail_increases_event_recall_both_SSP"]==3

def test_reject_unfrozen_ssp_fitting():
    a,b,c=fixture();b["ssp_fitted"]=True
    with pytest.raises(ValueError,match="Missing or altered"):compare(a,b,c)

def test_reject_incomplete_model_seed():
    a,b,c=fixture()
    a["scenarios"]["ssp245"]["models"]["fno"]["3"]["event_day_recall"]["individual"].pop()
    with pytest.raises(ValueError,match="Incomplete frozen-seed"):compare(a,b,c)

def test_reject_seed_mean_provenance_change():
    a,b,c=fixture()
    b["scenarios"]["ssp585"]["models"]["fno"]["0"]["event_day_recall"]["mean"]=.99
    with pytest.raises(ValueError,match="mean mismatch"):compare(a,b,c)

def test_reject_native_grid_mismatch():
    a,b,c=fixture();a["scenarios"]["ssp245"]["source"]["native_grid_shape"]=[23,23]
    with pytest.raises(ValueError,match="native region"):compare(a,b,c)

def test_explicit_counterexample_kept_not_suppressed():
    a,b,c=fixture()
    z=c["scenarios"]["ssp245"]["models"]["fno"]["3"]["event_day_recall"]
    z["individual"]=[.4,.4,.4];z["mean"]=.4
    r=compare(a,b,c)
    q=next(x for x in r["scenario_findings"] if x["gcm"]=="CAMS" and x["scenario"]=="ssp245")
    assert q["strong_tail_minus_standard_event_recall_pp"]<0
    assert r["GCM_level_counts"]["strong_tail_increases_event_recall_both_SSP"]==2

def test_write_four_evidence_files_without_overwrite(tmp_path):
    d=compare(*fixture());out=tmp_path/"three_GCM"
    export(d,out)
    assert len(list(out.glob("*.csv")))==3
    assert len(list(csv.DictReader((out/"v525_six_GCM_SSP_event_findings.csv").open())))==6
    assert "three independent GCMs" in (out/"v525_three_GCM_contiguous_event_results.md").read_text()
    with pytest.raises(FileExistsError):export(d,out)
