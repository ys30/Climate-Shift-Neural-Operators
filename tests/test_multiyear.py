import numpy as np
import xarray as xr
import pytest
from scripts.export_wb2_multiyear import extract

def test_monthly_sampling_and_missing_month_logs():
    times = np.arange(np.datetime64("2000-01-01"),
                      np.datetime64("2000-02-01"), np.timedelta64(6, "h"))
    ds = xr.Dataset(
        {"2m_temperature": (("time", "latitude", "longitude"),
                            np.ones((len(times), 3, 3)))},
        coords={"time": times, "latitude": [30., 35., 40.],
                "longitude": [235., 240., 245.]}
    )
    x, y, years, stamps, missing = extract(
        ds, [2000], max_per_month=2, stride=4)
    assert x.shape == (2, 1, 3, 3)
    assert (years == 2000).all()
    assert len(missing) == 11
    assert len(np.unique(stamps)) == len(stamps)
    assert stamps[0] == np.datetime64("2000-01-02T00")
    assert stamps[-1] == np.datetime64("2000-01-31T00"), "uniform sampling must cover month end"
    with pytest.raises(ValueError):
        extract(ds, [2000, 2000])
