"""v5.24 AWI continuous 2050 SSP event-day parity with BCC and CAMS."""
import numpy as np
import pytest
from scripts.evaluate_cmip6_v524_awi_contiguous_summer import STATUS,score,evaluate
from scripts.evaluate_cmip6_v520_cams_contiguous_summer import dates,verify_dates,run_mask

def test_same_91_day_summer_as_other_GCMs():
    d=dates();verify_dates(d)
    assert len(d)==91 and d[0]==(2050,6,1) and d[-1]==(2050,8,30)

def test_shared_event_rule_marks_3_or_more_days_only():
    hot=np.zeros((91,26,27),dtype=bool)
    hot[20:23,4,5]=True
    hot[35:37,4,6]=True
    found=run_mask(hot)
    assert int(found.sum())==3 and found[20:23,4,5].all()

def test_awi_month_local_P95_and_event_metrics_match_grid():
    frozen={"latitude":np.linspace(25.5,49.5,26),
            "local_p95":np.full((12,26,27),300.,dtype=np.float32)}
    X=np.full((91,1,26,27),295.,dtype=np.float32)
    Y=X.copy();Y[4:9,0,3:5,4:6]=310.
    months=np.array([6]*29+[7]*31+[8]*31)
    result=score(X,Y,months,frozen)
    assert result["rmse_K"]>0
    assert result["event_day_recall"]==0.
    assert result["observed_3day_pixel_windows"]==3*4

def test_reject_missing_frozen_AWI_models_before_catalog():
    with pytest.raises(ValueError,match="Incomplete pre-SSP"):
        evaluate({"models":{}},{})

def test_status_distinct_from_other_GCMs():
    assert "AWI" in STATUS and "BCC" not in STATUS and "CAMS" not in STATUS
