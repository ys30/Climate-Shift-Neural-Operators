"""E1 v5.9 source-only Kelvin anomaly and source-cohort guards."""
import numpy as np
import pytest
from scripts.audit_cmip6_v59_source_warming import weighted_mean,compare

def data():
    hist=np.full((240,1,2,3),285.0)
    fut=np.concatenate([np.full((24,1,2,3),286.0),
                        np.full((24,1,2,3),287.0),
                        np.full((24,1,2,3),289.0)])
    return hist,np.ones(240,dtype=bool),fut,np.array([30.0,40.0]),np.repeat([2030,2050,2080],24)

def test_sampled_change_identity():
    r=compare(*data())
    assert len(r)==3
    assert [x["sampled_change_vs_historical_K"] for x in r]==pytest.approx([1,2,4])
    assert [x["spatial_anomaly_p50_K"] for x in r]==pytest.approx([1,2,4])

def test_latitude_weighting():
    x=weighted_mean(np.array([[1.0,1.0],[4.0,4.0]]),np.array([30.0,60.0]))
    assert 1.0<x<2.5

def test_missing_future_date_rejected():
    h,m,f,l,y=data()
    with pytest.raises(ValueError,match="24 future"):
        compare(h,m,f[:-1],l,y[:-1])

def test_historical_count_rejected():
    h,m,f,l,y=data()
    m[0]=False
    with pytest.raises(ValueError,match="240 Historical"):
        compare(h,m,f,l,y)

def test_nonfinite_rejected():
    h,m,f,l,y=data()
    f[0,0,0,0]=float("nan")
    with pytest.raises(ValueError,match="Nonfinite"):
        compare(h,m,f,l,y)

def test_native_grid_mismatch_rejected():
    h,m,f,l,y=data()
    with pytest.raises(ValueError,match="Native grids"):
        compare(h,m,f[:,:,:,:2],l,y)
