import numpy as np
import pytest
from climate_shift.extremes import fit_hot_threshold, hot_diagnostics


def test_threshold_fit_on_training_only():
    train = np.arange(20, dtype=float).reshape(5, 1, 2, 2)
    assert fit_hot_threshold(train, 0.90) == pytest.approx(17.1)


def test_tail_errors_and_latitude_weights():
    ref = np.array([[[[2., 4.], [10., 12.]]]])
    pred = np.array([[[[2., 5.], [8., 13.]]]])
    out = hot_diagnostics(pred, ref, 9, latitude=np.array([0., 60.]))
    assert out["hot_reference_count"] == 2
    assert out["hot_recall"] == pytest.approx(0.5)
    assert out["hot_conditional_bias"] == pytest.approx(-0.5)
    assert out["frequency_bias"] == pytest.approx(-0.5 / 3)


def test_absent_events_are_undefined_not_perfect():
    x = np.zeros((2, 1, 2, 2))
    out = hot_diagnostics(x, x, 5)
    assert np.isnan(out["hot_recall"])
    assert np.isnan(out["hot_conditional_bias"])


def test_invalid_threshold_grid_or_latitudes():
    with pytest.raises(ValueError):
        fit_hot_threshold(np.ones((2, 1, 2, 2)), 1)
    with pytest.raises(ValueError):
        hot_diagnostics(np.ones((2, 1, 2, 2)), np.ones((2, 1, 2, 2)), 1, latitude=np.array([0]))
