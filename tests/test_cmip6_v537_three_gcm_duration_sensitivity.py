"""v5.37 three original GCM archived 2/3/5-day sensitivity integrity tests."""
from copy import deepcopy
import csv
import pytest
from scripts.audit_cmip6_v537_three_gcm_duration_sensitivity import audit,write,SOURCES,STATUS

def sample():
    result={}
    for g,(_,run) in SOURCES.items():
        records=[]
        for year in (2040,2050):
            for scenario in ("ssp245","ssp585"):
                for days in (2,3,5):
                    for model,weight in (("persistence",None),("fno",1),("fno",4)):
                        rec=.80 if model=="persistence" else (.60 if weight==1 else .70)
                        pre=.90 if model=="persistence" else (.85 if weight==1 else .80)
                        records.append(dict(year=year,scenario=scenario,min_days=days,
                           model=model,historical_multiplier=weight,
                           event_day_recall=rec,event_day_precision=pre,
                           observed_event_day_area_weight=float(100-days),
                           predicted_event_day_area_weight=float(100-days)))
        result[g]={
          "status":SOURCES[g][0],"years":[2040,2050],
          "scenarios":["ssp245","ssp585"],"durations":[2,3,5],
          "ssp_fitted":False,"all_original_three_day_reference_scores_reproduced":True,
          "n_original_frozen_checkpoints":18,"independent_GCM_count":1,
          "checkpoint_sha":{str(i):str(i) for i in range(18)},
          "records":records}
    return result

def test_all_108_rows_and_36_duration_cases():
    r=audit(sample())
    assert r["status"]==STATUS and r["independent_GCM_count"]==3
    assert len(r["all_108_source_rows"])==108
    assert len(r["scenario_duration_findings"])==36
    assert len(r["paired_two_vs_five_day_changes"])==12
    assert all(z["valid_scenarios"]==12 for z in r["duration_counts"].values())

def test_missing_duration_case_rejected():
    s=sample()
    s["BCC"]["records"].pop()
    with pytest.raises(ValueError,match="Missing/duplicate"):audit(s)

def test_non_decreasing_observed_event_area_rejected():
    s=sample()
    for r in s["AWI"]["records"]:
        if r["min_days"]==5:r["observed_event_day_area_weight"]=105.0
    with pytest.raises(ValueError,match="shrink"):audit(s)

def test_requires_archive_three_day_reproduction():
    s=sample();s["CAMS"]["all_original_three_day_reference_scores_reproduced"]=False
    with pytest.raises(ValueError,match="incomplete or changed"):audit(s)

def test_outputs_keep_nonindependent_units(tmp_path):
    d=audit(sample());p=tmp_path/"report"
    write(d,p)
    with (p/"v537_36_duration_cases.csv").open() as f:
        assert len(list(csv.DictReader(f)))==36
    assert "three GCM members are the only independent" in (p/"E1_v537_duration_sensitivity_manuscript_results.md").read_text()
    with pytest.raises(FileExistsError):write(d,p)
