import numpy as np
import pytest
from climate_shift.conditional_skill import fit_monthly_standardization, conditional_error


def _data():
    rng = np.random.default_rng(51)
    times = np.array([
        np.datetime64(f"2008-{m:02d}-{day:02d}", "ns")
        for m in range(1,13) for day in range(1,25)
    ])
    values = rng.normal(280, 2, size=(len(times),1,3,4)).astype("float32")
    return times, values


def test_temperature_anomaly_bins_and_train_only_reference():
    times, train = _data()
    mean, sd = fit_monthly_standardization(train, times)
    test_times = times[:4] + np.timedelta64(4, "Y") if False else np.array([
        np.datetime64(f"2012-01-{d:02d}", "ns") for d in range(1,5)
    ])
    y = train[:4].copy()
    y[0] += 12
    pred = y - 1
    out = conditional_error(pred, y, test_times, mean, sd, np.array([25.,35.,45.]))
    bins = out["bins"]
    assert sum(v["unweighted_gridcell_count"] for v in bins.values()) == y.shape[0]*3*4
    assert sum(v["weighted_fraction"] for v in bins.values()) == pytest.approx(1)
    assert bins["hot_gt_p2"]["unweighted_gridcell_count"] > 0
    for row in bins.values():
        if row["unweighted_gridcell_count"] > 0:
            assert row["weighted_rmse_K"] == pytest.approx(1)
            assert row["weighted_bias_K"] == pytest.approx(-1)


def test_fails_on_missing_climatology_month():
    times, train = _data()
    with pytest.raises(ValueError, match="Insufficient month"):
        fit_monthly_standardization(train[:23], times[:23])


def test_rejects_invalid_prediction_shape():
    times, train = _data()
    mean, sd = fit_monthly_standardization(train, times)
    with pytest.raises(ValueError, match="same-shape"):
        conditional_error(train[:3],train[:2],times[:2],mean,sd,np.array([25.,35.,45.]))
