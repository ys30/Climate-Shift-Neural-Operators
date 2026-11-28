"""E1 v5.34 AWI hot-run duration sensitivity and unchanged 3-day reference."""
import numpy as np
import pytest
from scripts.evaluate_cmip6_v534_awi_duration_sensitivity import (
    DURATIONS,YEARS,SSPS,event,check_match,reference_gate,summarize)

def test_duration_two_three_five_distinct():
    p95=np.full((12,2,3),300.)
    y=np.full((91,1,2,3),295.)
    y[10:14,0,0,0]=310.
    y[20:22,0,0,1]=310.
    p=y.copy()
    p[10:13,0,0,0]=309.
    p[13,0,0,0]=295.
    months=np.full(91,6,dtype=int)
    a={n:event(p,y,months,p95,np.array([35.,40.]),n) for n in DURATIONS}
    assert a[2]["event_day_recall"]==pytest.approx(5/6)
    assert a[3]["event_day_recall"]==pytest.approx(3/4)
    assert a[2]["event_day_recall"]>a[3]["event_day_recall"]
    assert a[5]["event_day_recall"] is None
    assert a[5]["observed_event_day_area_weight"]==0.

def test_identical_persistence_event_day_recall_is_one():
    p95=np.full((12,1,1),300.)
    x=np.full((91,1,1,1),295.)
    x[2:8]=310.
    v=event(x,x,np.full(91,6),p95,np.array([33.]),3)
    assert v["event_day_recall"]==pytest.approx(1.)
    assert v["event_day_precision"]==pytest.approx(1.)

def test_reference_sha_drift_rejected():
    with pytest.raises(ValueError,match="changed"):
        reference_gate({"year":2040,"ssp_fitted":False,"checkpoint_sha":{"a":"wrong"},
                        "scenarios":{"ssp245":{},"ssp585":{}}},2040,{"a":"actual"})

def test_strict_reproduce_reference_gate():
    check_match(.50,.50000001,"safe",tolerance=1e-6)
    with pytest.raises(ValueError,match="differs"):
        check_match(.4,.5,"regr")

def test_three_seed_mean_protected():
    assert summarize([{"event_day_recall":.2},{"event_day_recall":.4},
         {"event_day_recall":.6}],"event_day_recall")==pytest.approx(.4)
    with pytest.raises(ValueError,match="Missing original seeds"):
        summarize([{"event_day_recall":.2}],"event_day_recall")

def test_declared_years_scenarios_durations():
    assert YEARS==(2040,2050)
    assert SSPS==("ssp245","ssp585")
    assert DURATIONS==(2,3,5)
