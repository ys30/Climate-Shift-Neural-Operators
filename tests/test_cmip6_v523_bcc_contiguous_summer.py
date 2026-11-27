"""v5.23 fixed BCC 2050 native-no-leap contiguous summer event pilot."""
import numpy as np
import pytest
from scripts.evaluate_cmip6_v523_bcc_contiguous_summer import score,evaluate,predict,STATUS
from scripts.evaluate_cmip6_v520_cams_contiguous_summer import dates,verify_dates

def test_same_summer_cohort_as_original_CAMS():
    schedule=dates()
    verify_dates(schedule)
    assert len(schedule)==91
    assert schedule[0]==(2050,6,1)
    assert schedule[-1]==(2050,8,30)

def test_identical_3day_definition_across_GCM():
    from scripts.evaluate_cmip6_v520_cams_contiguous_summer import run_mask
    hot=np.zeros((91,2,3),dtype=bool)
    hot[10:14,1,2]=True
    hot[20:22,0,1]=True
    e=run_mask(hot)
    assert e.sum()==4 and e[10:14,1,2].all()

def test_source_event_metrics_uses_bcc_native_grid():
    frozen={"lat":np.linspace(26.,48.,23),"p95":np.full((12,23,23),300.)}
    X=np.full((91,1,23,23),295.,dtype=np.float32)
    Y=X.copy()
    Y[5:9,:,5:8,5:8]=310.
    months=np.array([6]*29+[7]*31+[8]*31)
    z=score(X,Y,months,frozen)
    assert z["event_day_recall"]==0.
    assert z["rmse_K"]>0

def test_unverified_model_inventory_rejected():
    with pytest.raises(ValueError,match="Incomplete pre-SSP"):
        evaluate({"checkpoints":{}},{})

def test_output_status_is_not_CAMS():
    assert "BCC" in STATUS and "CAMS" not in STATUS
