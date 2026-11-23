"""Regression tests for explicit forecast-origin and target timestamp provenance."""
import numpy as np
import pytest
from climate_shift.data import load_npz_pairs
from climate_shift.audit_e1 import audit


def write_npz(path, sources, targets, years=None):
    if years is None:
        years = targets.astype("datetime64[Y]").astype(int) + 1970
    n = len(targets)
    np.savez_compressed(path, X=np.ones((n, 1, 8, 8), dtype="float32"),
                        y=np.ones((n, 1, 8, 8), dtype="float32"),
                        years=years, source_times=sources, target_times=targets)


def test_exact_24h_with_cross_year_origin(tmp_path):
    targets = np.array(["2010-01-01T00", "2010-01-03T00"], dtype="datetime64[ns]")
    sources = targets - np.timedelta64(24, "h")
    p = tmp_path / "valid.npz"
    write_npz(p, sources, targets)
    x, y, years = load_npz_pairs(p)
    assert years.tolist() == [2010, 2010]
    result = audit(p)
    assert result["exact_24h_pairs_verified"] is True
    assert result["first_source"].startswith("2009-12-31")


def test_wrong_lead_raises(tmp_path):
    targets = np.array(["2010-01-02T00", "2010-01-03T00"], dtype="datetime64[ns]")
    sources = targets - np.timedelta64(24, "h")
    sources[-1] = targets[-1] - np.timedelta64(18, "h")
    p = tmp_path / "wrong.npz"
    write_npz(p, sources, targets)
    with pytest.raises(ValueError, match="exact"):
        load_npz_pairs(p)


def test_source_requires_target(tmp_path):
    p = tmp_path / "missing.npz"
    np.savez_compressed(p, X=np.ones((1, 1, 8, 8)), y=np.ones((1, 1, 8, 8)),
                        years=np.array([2010]), source_times=np.array(["2009-12-31"], dtype="datetime64[ns]"))
    with pytest.raises(ValueError, match="require"):
        load_npz_pairs(p)
