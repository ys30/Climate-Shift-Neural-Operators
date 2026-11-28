"""E1 v5.40 true event episode diagnostics, NOT heatwave-object tracking."""
import numpy as np
import pytest
from scripts.evaluate_cmip6_v540_awi_episode_objects import (
    segments,match_one_pixel,diagnostic,audit_3day,analyze,MIN_DAYS)
def arrays():
    Y=np.full((91,1,2,2),295.,dtype=np.float32)
    P=Y.copy()
    p95=np.full((12,2,2),300.,dtype=float)
    months=np.full(91,6,dtype=int)
    lat=np.array([30.,40.])
    return P,Y,p95,months,lat

def test_true_episode_segments_are_half_open_and_keep_boundary_events():
    z=np.zeros(91,bool);z[:3]=1;z[10:15]=1;z[-4:]=1
    assert segments(z)==[(0,3),(10,15),(87,91)]
    assert segments(z,min_days=5)==[(10,15)]
    assert segments(np.zeros(91,bool))==[]

def test_nonoverlap_is_not_same_event_even_when_nearby():
    m=match_one_pixel([(10,15)],[(15,20)])
    assert m["matched_truth"]==0 and m["matched_pred"]==0

def test_one_to_one_and_fragmentation_from_split_forecasts():
    m=match_one_pixel([(10,21)],[(10,13),(15,19)])
    assert m["matched_truth"]==1 and m["matched_pred"]==1
    assert m["fragmented_truth"]==1
    assert len(m["pairs"])==1

def test_fragmentation_and_uncensored_duration_bias_excludes_edges():
    p,y,p95,months,lat=arrays()
    y[10:20,0,0,0]=310
    p[11:14,0,0,0]=310
    p[16:19,0,0,0]=310
    val=diagnostic(p,y,months,p95,lat)
    assert val["episode_recall"]==pytest.approx(1.)
    assert val["episode_precision"]==pytest.approx(.5)
    assert val["fragmentation_rate"]==pytest.approx(1.)
    assert val["mean_matched_interior_duration_bias_days"]==pytest.approx(-7.)
    assert val["mean_matched_interior_onset_lag_days"]==pytest.approx(1.)

def test_boundary_duration_not_misreported_as_complete_event():
    p,y,p95,months,lat=arrays()
    y[:6,0,0,0]=310;p[:6,0,0,0]=310
    d=diagnostic(p,y,months,p95,lat)
    assert d["episode_recall"]==pytest.approx(1.)
    assert d["area_weighted_boundary_censored_true_episodes"]>0
    assert d["area_weighted_matched_interior_episodes"]==0
    assert d["mean_matched_interior_duration_bias_days"] is None

def test_original_event_day_reference_gate_detects_drift():
    p,y,p95,months,lat=arrays()
    y[10:14,0,0,0]=310
    p[10:13,0,0,0]=310
    from scripts.evaluate_cmip6_v534_awi_duration_sensitivity import event
    ref=event(p,y,months,p95,lat,3)
    audit_3day(p,y,months,p95,lat,ref,"orig")
    with pytest.raises(ValueError,match="differs"):
        audit_3day(p,y,months,p95,lat,{**ref,"event_day_recall":.01},"changed")

def test_original_18_frozen_models_required_before_any_ssp_access():
    with pytest.raises(ValueError,match="18 original"):
        analyze({"models":{},"checkpoint_sha256_by_arm":{}},{},{},{})
