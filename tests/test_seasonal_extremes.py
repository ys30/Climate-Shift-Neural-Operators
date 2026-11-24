import numpy as np
import pytest

from climate_shift.seasonal_extremes import (
    fit_local_monthly_p95, local_monthly_hot_diagnostics,
)


def synthetic_train():
    dates = []
    data = []
    for month in range(1, 13):
        for day in range(1, 25):
            dates.append(np.datetime64(f"2008-{month:02d}-{day:02d}", "ns"))
            # Distinct local temperature climatologies; cell-level warm anomalies.
            data.append(np.array([[[280 + month + day / 10, 285 + month + day / 10],
                                   [290 + month + day / 10, 295 + month + day / 10]]]))
    return np.asarray(data, dtype="float32"), np.asarray(dates)


def test_thresholds_are_month_specific_and_spatially_local():
    train, dates = synthetic_train()
    threshold, meta = fit_local_monthly_p95(train, dates)
    assert threshold.shape == (12, 2, 2)
    assert threshold[5, 0, 0] > threshold[0, 0, 0]
    assert threshold[0, 1, 1] > threshold[0, 0, 0]
    assert meta["actual_samples_per_month"]["01"] == 24
    assert meta["fitting_scope"] == "training_targets_only_monthly_local_gridcell"


def test_test_data_cannot_refit_training_threshold():
    train, dates = synthetic_train()
    threshold, _ = fit_local_monthly_p95(train, dates)
    heldout = np.array([[[[threshold[0, 0, 0] + 5, threshold[0, 0, 1] + 5],
                         [threshold[0, 1, 0] + 5, threshold[0, 1, 1] + 5]]]])
    target_times = np.array(["2012-01-07"], dtype="datetime64[ns]")
    result = local_monthly_hot_diagnostics(
        heldout - 2, heldout, target_times, threshold, latitude=np.array([25., 50.]))
    assert result["hot_recall"] == pytest.approx(1.)
    assert result["hot_conditional_bias"] == pytest.approx(-2.)
    assert result["weighting"] == "cos(latitude)"
    np.testing.assert_allclose(
        threshold[0], fit_local_monthly_p95(train, dates)[0][0])


def test_reject_missing_month_and_malformed_dates():
    train, dates = synthetic_train()
    with pytest.raises(ValueError, match="Insufficient"):
        fit_local_monthly_p95(train[:24], dates[:24])
    with pytest.raises(ValueError, match="timestamps"):
        fit_local_monthly_p95(train, np.array(["not-a-date"] * len(train), dtype="U16"))


def test_undefined_hot_event_metrics_are_null():
    train, dates = synthetic_train()
    threshold, _ = fit_local_monthly_p95(train, dates)
    ref = np.zeros((1, 1, 2, 2))
    report = local_monthly_hot_diagnostics(
        ref, ref, np.array(["2012-01-07"], dtype="datetime64[ns]"), threshold)
    assert report["hot_recall"] is None
    assert report["hot_precision"] is None
    assert report["hot_reference_count"] == 0


def test_monthly_thresholds_use_calendar_month_not_day_of_year():
    train, dates = synthetic_train()
    q, _ = fit_local_monthly_p95(train, dates)
    january = np.ones((1, 1, 2, 2)) * (q[0] + q[1]) / 2
    report = local_monthly_hot_diagnostics(
        january, january, np.array(["2012-01-15"], dtype="datetime64[ns]"), q)
    assert report["hot_recall"] == pytest.approx(1.)
