"""E1 v5.18 validate third-GCM SSP checks and all frozen metrics."""
import numpy as np
import pytest
from scripts.evaluate_cmip6_v518_cams_frozen_ssp import dates,aggregate,manifest_for_CAMS

def frozen():
    return {"catalog_sha":"catalog123","member":"r2i1p1f1","grid":"gn"}
def manifest():
    row={"source_id":"CAMS-CSM1-0","member_id":"r2i1p1f1","grid_label":"gn",
        **{s:{"store":f"gs://cmip6/CMIP6/ScenarioMIP/CAMS/CAMS-CSM1-0/{s}/r2i1p1f1/day/tas/gn/v1/"} for s in ("ssp245","ssp585")}}
    return {"status":"catalog_metadata_only_no_climate_fields_or_transfer_skill",
        "catalog_sha256":"catalog123","selected_candidates":[row]}

def test_schedule_72_distinct_native_source_dates():
    d=dates()
    assert len(d)==len(set(d))==72
    assert {x[0] for x in d}=={2030,2050,2080}
    assert {x[2] for x in d}=={8,20}

def test_exact_source_identity():
    assert manifest_for_CAMS(manifest(),frozen())["source_id"]=="CAMS-CSM1-0"

def test_reject_changed_catalog_sha():
    d=manifest();d["catalog_sha256"]="different"
    with pytest.raises(ValueError,match="Catalog changed"):manifest_for_CAMS(d,frozen())

def test_reject_mismatched_SSP_source():
    d=manifest()
    d["selected_candidates"][0]["ssp245"]["store"]="gs://cmip6/invalid"
    with pytest.raises(ValueError,match="Wrong future"):manifest_for_CAMS(d,frozen())

def test_aggregate_three_predeclared_seeds():
    x={"weighted":{"rmse":2.0,"mae":1.3,"bias":.1},
       "monthly_hot":{"hot_recall":.8,"hot_precision":.7,"predicted_hot_fraction":.05}}
    d=aggregate([x,x,x])
    assert d["weighted_rmse_K"]["mean"]==pytest.approx(2.)
    assert d["local_p95_recall"]["individual"]==[.8,.8,.8]

def test_reject_missing_seed():
    x={"weighted":{"rmse":2.},"monthly_hot":{}}
    with pytest.raises(ValueError,match="Three frozen"):aggregate([x,x])

def test_reject_nonfinite_scores():
    x={"weighted":{"rmse":2.,"mae":1.,"bias":0.},
       "monthly_hot":{"hot_recall":np.nan,"hot_precision":.7,"predicted_hot_fraction":.05}}
    with pytest.raises(ValueError,match="Invalid CAMS"):aggregate([x,x,x])
