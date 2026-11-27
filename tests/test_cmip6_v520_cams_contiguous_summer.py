"""E1 v5.20 contiguous 2050 native calendar and heat event day rules."""
import numpy as np
import pytest
from scripts.evaluate_cmip6_v520_cams_contiguous_summer import (
    dates,verify_dates,run_mask,event_stats,summarize_seed_values)

def test_summer_91_real_consecutive_dates():
    d=dates()
    assert len(d)==91
    assert d[0]==(2050,6,1) and d[-1]==(2050,8,30)
    verify_dates(d)

def test_run_mask_excludes_short_events_includes_full_long_events():
    hot=np.zeros((12,2,3),bool)
    hot[1:3,0,0]=True   # 2 days, not a qualifying event
    hot[4:8,0,0]=True   # four days, all four should be marked
    hot[8:11,1,2]=True  # exact 3 days
    event=run_mask(hot)
    assert event[:,0,0].sum()==4
    assert event[:,1,2].sum()==3
    assert not event[1:3,0,0].any()

def test_nonconsecutive_three_hot_days_are_not_heat_events():
    hot=np.zeros((10,1,1),bool)
    hot[[1,3,5]]=True
    assert not run_mask(hot).any()

def test_exact_pixel_event_recall_precision():
    p95=np.full((12,2,3),300.0,dtype="float32")
    true=np.full((91,1,2,3),290.,dtype="float32")
    true[15:19,0,0,0]=310. # four true event days
    pred=true.copy()
    pred[15,0,0,0]=280. # only days 16-18 exceed; 3/4 true event-day recall
    months=np.array([6]*29+[7]*31+[8]*31)
    score=event_stats(pred,true,months,p95,np.array([30.,45.]))
    assert score["event_day_recall"]==pytest.approx(.75)
    assert score["event_day_precision"]==pytest.approx(1.)
    assert score["observed_3day_pixel_windows"]==2
    assert score["forecast_3day_pixel_windows"]==1

def test_no_true_events_not_fabricated_as_recall_one():
    p95=np.full((12,2,3),300.)
    target=np.full((91,1,2,3),290.)
    months=np.array([6]*29+[7]*31+[8]*31)
    r=event_stats(target,target,months,p95,np.array([32.,45.]))
    assert r["event_day_recall"] is None and r["event_day_precision"] is None

def test_missing_seed_rejected():
    with pytest.raises(ValueError,match="3 frozen"):
        summarize_seed_values([{},{}])

def test_null_seed_metrics_are_all_explicitly_undefined_or_all_valid():
    example={"rmse_K":2.,"mae_K":1.,"bias_K":0.,
      "daily_p95_recall":.6,"daily_p95_precision":.8,
      "event_day_recall":None,"event_day_precision":None}
    output=summarize_seed_values([example]*3)
    assert output["event_day_recall"]["mean"] is None
    assert output["rmse_K"]["mean"]==pytest.approx(2.)
    broken=dict(example,event_day_recall=.8)
    with pytest.raises(ValueError,match="Missing or nonfinite"):
        summarize_seed_values([example,example,broken])
