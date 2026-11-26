"""Fixed Historical only 3-seed loss-arm aggregation must not cherry pick seeds."""
import copy
import pytest
from scripts.run_cmip6_v47_historical_arms import aggregate,REQUIRED_SHA


def fixture(weight=1.0):
    reports=[]
    for seed in (11,22,33):
        m={}
        for name in ("fno","unet"):
            m[name]={
                "parameter_count":1234,
                "checkpoint":name+".pt",
                "checkpoint_sha256":"a"*64,
                "best_epoch":60,
                "validation_weighted":{"rmse":2.2,"mae":1.8,"bias":-0.2},
                "validation_local_p95":{
                    "hot_recall":0.5,"hot_precision":0.6,
                    "predicted_hot_fraction":0.08},
            }
        reports.append({
            "seed":seed,"npz_sha256":REQUIRED_SHA,"tail_weight":weight,
            "epochs":100,"ssp_data_read":False,
            "validation_persistence":{"rmse":2.3298,"mae":1.5,"bias":0.0},
            "models":copy.deepcopy(m),
        })
    return reports


def test_all_three_seeds_and_all_model_metrics():
    out=aggregate(fixture(),1.0)
    assert out["seed_ids"]==[11,22,33]
    assert out["models"]["fno"]["weighted_rmse_K"]["mean"]==2.2
    assert len(out["models"]["unet"]["checkpoints"])==3
    assert out["ssp_data_read"] is False


def test_reject_missing_seed_or_changed_data():
    with pytest.raises(ValueError,match="Unfrozen"):
        aggregate(fixture()[:2],1.0)
    reports=fixture()
    reports[1]["npz_sha256"]="different"
    with pytest.raises(ValueError,match="Historical dataset"):
        aggregate(reports,1.0)


def test_reject_posthoc_weight_and_ssp_read():
    with pytest.raises(ValueError,match="Unfrozen"):
        aggregate(fixture(),2.0)
    reports=fixture()
    reports[1]["ssp_data_read"]=True
    with pytest.raises(ValueError,match="Historical dataset"):
        aggregate(reports,1.0)


def test_reject_changed_baseline():
    reports=fixture()
    reports[0]["validation_persistence"]["rmse"]=0.5
    with pytest.raises(ValueError,match="Unmatched Persistence"):
        aggregate(reports,1.0)
