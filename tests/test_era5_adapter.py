import numpy as np
import pytest
import xarray as xr
from climate_shift.era5_adapter import make_24h_pairs


def fixture_ds(missing=False):
    time = np.arange(np.datetime64("2019-07-01T00"), np.datetime64("2019-07-04T00"), np.timedelta64(6, "h"))
    if missing:
        time = np.delete(time, 1)
    latitude = np.array([20., 30., 40., 50., 60.])
    longitude = np.array([230., 240., 250., 260., 270.])
    values = np.arange(len(time)*25, dtype=float).reshape(len(time), 5, 5) + 273.15
    return xr.Dataset({"2m_temperature": (("time", "latitude", "longitude"), values)},
                      coords={"time": time, "latitude": latitude, "longitude": longitude})


def test_real_time_alignment_and_360_longitudes():
    ds = fixture_ds()
    x, y, years, targets = make_24h_pairs(ds, start="2019-07-01", end="2019-07-03", stride=1, max_samples=2)
    assert x.shape == (2, 1, 3, 3)
    assert years.tolist() == [2019, 2019]
    assert targets[0] - ds.time.values[0] == np.timedelta64(24, "h")
    np.testing.assert_allclose(y[0] - x[0], 4 * 25)


def test_missing_time_is_not_index_shifted():
    ds = fixture_ds(missing=True)
    x, y, years, targets = make_24h_pairs(ds, start="2019-07-01", end="2019-07-03", stride=1, max_samples=2)
    assert targets[0] == np.datetime64("2019-07-02T00")
    # First source time is 2019-07-01T00, target 24h later; other missing timestamps must not perturb alignment.
    assert len(x) == 2


def test_reject_duplicate_time():
    ds = fixture_ds()
    ds = ds.assign_coords(time=np.array([ds.time.values[0]] * ds.sizes["time"]))
    with pytest.raises(ValueError):
        make_24h_pairs(ds)


def test_reject_unavailable_area():
    with pytest.raises(ValueError):
        make_24h_pairs(fixture_ds(), lon_bounds=(-50., -40.))


def test_source_coordinates_are_from_actual_dataset():
    ds = fixture_ds(missing=True)
    x, y, years, targets, origins = make_24h_pairs(
        ds, start="2019-07-01", end="2019-07-03",
        stride=1, max_samples=3, return_source_times=True)
    assert len(origins) == len(targets)
    assert np.all(targets - origins == np.timedelta64(24, "h"))
    assert np.all(np.isin(origins, ds.time.values))
    assert np.all(np.isin(targets, ds.time.values))
