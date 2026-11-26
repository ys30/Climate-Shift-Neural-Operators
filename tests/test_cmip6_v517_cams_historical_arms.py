"""E1 v5.17: exact CAMS historical-only provenance and frozen settings."""
import json
import hashlib
import numpy as np
import pytest
from scripts.run_cmip6_v517_cams_historical_arms import validate_source,train_arm,EPOCHS,WEIGHTS
from scripts.export_cmip6_bcc_v51_historical import schedule

def sample(tmp_path):
    plan=schedule()
    train=np.array([r[0]=="train" for r in plan])
    yrs=np.array([r[1] for r in plan],dtype=np.int32)
    months=np.array([r[2] for r in plan],dtype=np.int8)
    rng=np.random.default_rng(4)
    x=(280+rng.normal(0,3,(1140,1,5,6))).astype("float32")
    y=(x+0.1*rng.normal(size=x.shape)).astype("float32")
    p95=np.stack([np.quantile(y[train & (months==m),0],.95,axis=0)
                  for m in range(1,13)]).astype("float32")
    dates=np.array([r[1]*366+r[2]*30+r[3] for r in plan],dtype=float)
    file=tmp_path/"cams.npz"
    np.savez_compressed(file,X=x,y=y,years=yrs,months=months,
        latitude=np.linspace(27,47,5),longitude_east=np.linspace(236,259,6),
        train_mask=train,source_native_time=np.array(["a"]*1140),
        target_native_time=np.array(["b"]*1140),
        source_cf_day_value=dates,target_cf_day_value=dates+1,
        monthly_train_p95_K=p95)
    meta={"status":"v516_CAMS_native_historical_only_1140_pairs_training_ready",
        "npz_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
        "source_id":"CAMS-CSM1-0","ssp_data_opened":False,
        "model_trained":False,"train_pairs":900,"validation_pairs":240,
        "train_only_monthly_p95":True,"native_grid_shape":[5,6],
        "native_calendar":"365_day"}
    m=tmp_path/"cams.json";m.write_text(json.dumps(meta))
    return file,m

def test_historical_only_source_and_derived_normalizer(tmp_path):
    path,meta=sample(tmp_path)
    d=validate_source(path,meta)
    assert d["shape"]==(5,6) and d["calendar"]=="365_day"
    assert d["mu"]>260 and d["sd"]>0
    assert d["train"].sum()==900

def test_reject_npz_hash_change(tmp_path):
    path,meta=sample(tmp_path)
    j=json.loads(meta.read_text());j["npz_sha256"]="0"*64
    meta.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="checksum"):validate_source(path,meta)

def test_reject_ssp_training_leak(tmp_path):
    path,meta=sample(tmp_path)
    j=json.loads(meta.read_text());j["ssp_data_opened"]=True
    meta.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="provenance"):validate_source(path,meta)

def test_reject_modified_training_threshold(tmp_path):
    path,meta=sample(tmp_path)
    with np.load(path,allow_pickle=False) as arr:
        data={k:arr[k] for k in arr.files}
    data["monthly_train_p95_K"]=data["monthly_train_p95_K"]+10.
    np.savez_compressed(path,**data)
    j=json.loads(meta.read_text());j["npz_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    meta.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="P95"):validate_source(path,meta)

def test_reject_unfrozen_weight_or_epoch(tmp_path):
    with pytest.raises(ValueError,match="preregistered"):
        train_arm({},tmp_path/"out",2)
    with pytest.raises(ValueError,match="preregistered"):
        train_arm({},tmp_path/"out",0,epochs=2)
