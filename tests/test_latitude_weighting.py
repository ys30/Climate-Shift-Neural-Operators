import numpy as np
import pytest
from climate_shift.metrics import latitude_weighted_metrics, regression_metrics
from climate_shift.compare_e1 import compare


def test_weighted_matches_unweighted_if_latitude_constant():
    pred = np.array([[[[2., 4.], [6., 8.]]]])
    ref = np.zeros_like(pred)
    assert latitude_weighted_metrics(pred, ref, np.array([30., 30.])) == pytest.approx(
        regression_metrics(pred, ref))


def test_weighted_downweights_high_latitude_error():
    reference = np.zeros((1, 1, 2, 2))
    prediction = np.array([[[[0., 0.], [10., 10.]]]])
    result = latitude_weighted_metrics(prediction, reference, np.array([0., 60.]))
    assert result["mae"] == pytest.approx(10 / 3)
    assert result["bias"] == pytest.approx(10 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(100 / 3))


def test_reject_bad_latitudes():
    x = np.ones((2, 1, 3, 2))
    with pytest.raises(ValueError, match="Latitude"):
        latitude_weighted_metrics(x, x, [30., 40.])
    with pytest.raises(ValueError, match="Latitude"):
        latitude_weighted_metrics(x, x, [30., 40., 120.])


def test_compare_includes_weighted_metrics_for_latitude_npz(tmp_path):
    rng = np.random.default_rng(8)
    x = (285 + rng.normal(size=(9, 1, 8, 8))).astype("float32")
    y = (x + rng.normal(scale=0.5, size=x.shape)).astype("float32")
    path = tmp_path / "with_latitude.npz"
    np.savez_compressed(path, X=x, y=y, years=np.repeat([2000, 2001, 2002], 3),
                        latitude=np.linspace(25., 50., 8))
    result = compare(path, tmp_path / "results.json", 2000, 2001, 2002,
                     epochs=1, width=4, modes=2)
    assert result["area_weighting_status"] == "cos(latitude)"
    assert set(result["latitude_weighted_metrics"]) == {"fno", "unet", "persistence"}
    assert result["latitude_weighted_metrics"]["persistence"]["rmse"] > 0
