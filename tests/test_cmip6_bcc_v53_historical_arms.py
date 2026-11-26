"""BCC v5.3 full cohort must report all seeds and keep SSP unseen."""
import copy
import pytest
from scripts.run_cmip6_bcc_v53_historical_arms import aggregate,BCC_V51_SHA


def reports(weight=1.):
    out=[]
    for seed in (11,22,33):
        model={}
        for n in ("fno","unet"):
            model[n]={
                "parameter_count":12345,"best_epoch":75,
                "checkpoint":n+".pt","checkpoint_sha256":"a"*64,
                "validation_weighted":{"rmse":2.1,"mae":1.4,"bias":-0.2},
                "validation_monthly_hot":{"hot_recall":.5,
                                          "hot_precision":.6,
                                          "predicted_hot_fraction":.1},
            }
        out.append({"seed":seed,"source_npz_sha256":BCC_V51_SHA,
                    "epochs":100,"tail_weight":weight,"calendar":"noleap",
                    "grid_shape":[23,23],"ssp_data_read":False,
                    "persistence_validation":{"rmse":2.812,"mae":1.9},
                    "models":copy.deepcopy(model)})
    return out


def test_all_models_and_seeds_reported():
    r=aggregate(reports(),1.)
    assert set(r["models"])=={"fno","unet"}
    assert r["seeds"]==[11,22,33]
    assert len(r["models"]["fno"]["checkpoints"])==3
    assert r["models"]["unet"]["local_p95_recall"]["mean"]==.5
    assert r["ssp_data_read"] is False


def test_partial_seeds_rejected():
    with pytest.raises(ValueError,match="all frozen"):
        aggregate(reports()[:2],1.)


def test_changed_source_or_future_read_rejected():
    a=reports()
    a[0]["source_npz_sha256"]="changed"
    with pytest.raises(ValueError,match="provenance"):
        aggregate(a,1.)
    a=reports()
    a[0]["ssp_data_read"]=True
    with pytest.raises(ValueError,match="provenance"):
        aggregate(a,1.)


def test_unfrozen_weight_or_missing_metric_rejected():
    with pytest.raises(ValueError,match="all frozen"):
        aggregate(reports(),2.)
    a=reports()
    a[0]["models"]["fno"]["validation_monthly_hot"]["hot_precision"]=None
    with pytest.raises(ValueError,match="Invalid BCC validation metric"):
        aggregate(a,1.)
