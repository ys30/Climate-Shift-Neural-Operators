import numpy as np
import pytest

from climate_shift.metrics import regression_metrics, warming_response_error, exceedance_frequency
from climate_shift.splits import validate_chronological_splits, train_only_standardize


def test_year_split_and_embargo():
    validate_chronological_splits([2000, 2001], [2004], [2007], embargo_years=2)
    with pytest.raises(ValueError):
        validate_chronological_splits([2000, 2001], [2002], [2004], embargo_years=1)
    with pytest.raises(ValueError):
        validate_chronological_splits([2000, 2001], [2001], [2004])


def test_normalization_is_train_only():
    a, b, mean, std = train_only_standardize(np.array([0.0, 2.0]), np.array([100.0]))
    np.testing.assert_allclose(a, [-1.0, 1.0])
    np.testing.assert_allclose(b, [99.0])
    np.testing.assert_allclose(mean, [1.0])
    np.testing.assert_allclose(std, [1.0])


def test_metrics():
    m = regression_metrics(np.array([1.0, 3.0]), np.array([0.0, 1.0]))
    assert m["mae"] == 1.5
    assert m["bias"] == 1.5
    np.testing.assert_allclose(m["rmse"], np.sqrt(2.5))
    assert exceedance_frequency(np.array([1, 2, 3]), threshold=2) == pytest.approx(1/3)
    assert warming_response_error(np.array([0]), np.array([3]), np.array([0]), np.array([2])) == 1.0


def test_reject_nan():
    with pytest.raises(ValueError):
        regression_metrics(np.array([np.nan]), np.array([1.0]))
