"""CAMS v5.36 native 2/3/5-day frozen temporal diagnostic gates."""
import numpy as np
import pytest
from scripts.evaluate_cmip6_v536_cams_duration_sensitivity import STATUS,gate,analyze
from scripts.evaluate_cmip6_v534_awi_duration_sensitivity import event,DURATIONS,YEARS,SSPS

def test_original_event_metric_on_CAMS_native_grid():
    pred=np.full((91,1,3,4),290.)
    pred[10:15,0,1,2]=311.
    p95=np.full((12,3,4),300.)
    x=event(pred,pred,np.full(91,6),p95,np.array([30.,35.,40.]),5)
    assert x["event_day_recall"]==pytest.approx(1)
    assert x["event_day_precision"]==pytest.approx(1)

def test_no_future_read_without_18_frozen_Historical_checkpoints():
    with pytest.raises(ValueError,match="18 original CAMS"):
        analyze({"checkpoints":{}},{},{},{})

def test_reject_changed_Historical_checkpoint_sha():
    f={"checkpoints":{("fno",0,11):{"sha256":"correct"}}}
    r={"year":2040,"source_model":"CAMS-CSM1-0","ssp_fitted":False,
       "checkpoint_sha":{"fno_weight0_seed11":"wrong"},
       "scenarios":{"ssp245":{},"ssp585":{}}}
    with pytest.raises(ValueError,match="SHA mismatch"):
        gate(r,2040,f)

def test_duration_settings_and_paired_years():
    assert "CAMS" in STATUS
    assert YEARS==(2040,2050)
    assert SSPS==("ssp245","ssp585")
    assert DURATIONS==(2,3,5)
