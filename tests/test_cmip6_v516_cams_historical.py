"""v5.16 Historical-only CAMS cohort source and time-leak guards."""
import copy
import numpy as np
import pytest
from scripts.export_cmip6_v516_cams_historical import validate_gate,verify_npz
from scripts.export_cmip6_bcc_v51_historical import schedule,validate_schedule,monthly_training_reference

def gate():
    e={k:{"status":"actual_bcc_native_one_day_pair_sanity_only",
          "store":f"gs://cmip6/CMIP6/CMIP/CAMS/CAMS-CSM1-0/{k}/r2i1p1f1/day/tas/gn/",
          "calendar":"365_day","grid_shape":[12,13],
          "latitude_sha256":"lat","longitude_sha256":"lon",
          "pairs":[{"finite_values":True},{"finite_values":True}]}
       for k in ("historical","ssp245","ssp585")}
    return {"status":"v515_CAMS_native_western_US_regional_one_day_real_source_gate",
        "model":"CAMS-CSM1-0","no_neural_training":True,
        "native_regional_shape":[12,13],"experiments":e}

def test_exact_1140_frozen_source_dates():
    plan=schedule()
    validate_schedule(plan)
    assert len(plan)==1140
    assert sum(r[0]=="train" for r in plan)==900
    assert sum(r[0]=="validation" for r in plan)==240

def test_actual_native_source_gate():
    shape,store,meta=validate_gate(gate())
    assert shape==(12,13)
    assert "/historical/" in store

def test_source_provenance_mismatch_rejected():
    g=gate();g["experiments"]["ssp585"]["calendar"]="360_day"
    with pytest.raises(ValueError,match="geometry"):validate_gate(g)

def test_wrong_source_store_rejected():
    g=gate();g["experiments"]["historical"]["store"]="gs://cmip6/fake/ssp585/day/tas/"
    with pytest.raises(ValueError,match="Historical-only"):validate_gate(g)

def test_local_p95_never_fit_on_validation():
    y=np.full((1140,1,4,5),280.0,dtype=np.float32)
    plan=schedule()
    yr=np.array([r[1] for r in plan])
    mo=np.array([r[2] for r in plan])
    train=np.array([r[0]=="train" for r in plan])
    y[~train]+=50.0
    p95,count=monthly_training_reference(y,yr,mo,train)
    assert np.allclose(p95,280.0)
    assert set(count.values())=={75}

def test_npz_rejects_incorrect_native_daily_lead(tmp_path):
    path=tmp_path/"bad.npz"
    x=np.full((1140,1,4,5),280.0,dtype=np.float32)
    years=np.array([r[1] for r in schedule()],dtype=np.int32)
    months=np.array([r[2] for r in schedule()],dtype=np.int8)
    train=np.array([r[0]=="train" for r in schedule()])
    start=np.arange(1140,dtype=float)
    np.savez_compressed(path,X=x,y=x,years=years,months=months,
        latitude=np.array([30.,35.,40.,45.]),
        longitude_east=np.array([235.,240.,245.,250.,255.]),
        train_mask=train,source_native_time=np.array(["a"]*1140),
        target_native_time=np.array(["b"]*1140),
        source_cf_day_value=start,target_cf_day_value=start+2,
        monthly_train_p95_K=np.full((12,4,5),280.0))
    with pytest.raises(ValueError,match="one day|day pairing|Non-native"):
        verify_npz(path,(4,5))
