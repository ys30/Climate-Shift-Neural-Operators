"""Fold-boundary guards: require strict separation of real predictor/target times."""
import numpy as np
import pytest
from climate_shift.split_isolation import audit_split_isolation


def make_pairfile(tmp_path, target_dates, include_sources=True):
    path = tmp_path / "pairs.npz"
    targets = np.asarray(target_dates, dtype="datetime64[ns]")
    data = {
        "X": np.ones((len(targets), 1, 8, 8), dtype="float32"),
        "y": np.ones((len(targets), 1, 8, 8), dtype="float32"),
        "years": targets.astype("datetime64[Y]").astype(int) + 1970,
        "target_times": targets,
    }
    if include_sources:
        data["source_times"] = targets - np.timedelta64(24, "h")
    np.savez_compressed(path, **data)
    return path


def test_disjoint_three_fold_data_pass(tmp_path):
    path = make_pairfile(tmp_path, ["2009-12-30", "2010-12-30", "2011-01-03"])
    masks = (np.array([1, 0, 0], bool), np.array([0, 1, 0], bool), np.array([0, 0, 1], bool))
    report = audit_split_isolation(path, masks, require_pair_times=True)
    assert report["status"] == "passed"
    assert report["boundaries"][0]["gap_hours"] > 0


def test_shared_target_and_next_source_rejected(tmp_path):
    path = make_pairfile(tmp_path, ["2010-12-31", "2011-01-01", "2012-01-04"])
    masks = (np.array([1, 0, 0], bool), np.array([0, 1, 0], bool), np.array([0, 0, 1], bool))
    with pytest.raises(ValueError, match="Boundary contamination"):
        audit_split_isolation(path, masks, require_pair_times=True)


def test_legacy_file_allowed_only_without_strict_requirement(tmp_path):
    path = make_pairfile(tmp_path, ["2010-01-01", "2011-01-01", "2012-01-01"], include_sources=False)
    masks = (np.array([1, 0, 0], bool), np.array([0, 1, 0], bool), np.array([0, 0, 1], bool))
    assert audit_split_isolation(path, masks)["status"] == "unverified_legacy_no_pair_times"
    with pytest.raises(ValueError, match="source_times"):
        audit_split_isolation(path, masks, require_pair_times=True)


def test_duplicate_fold_membership_rejected(tmp_path):
    path = make_pairfile(tmp_path, ["2010-01-01", "2011-01-01", "2012-01-01"])
    masks = (np.array([1, 1, 0], bool), np.array([0, 1, 0], bool), np.array([0, 0, 1], bool))
    with pytest.raises(ValueError, match="Overlapping"):
        audit_split_isolation(path, masks, require_pair_times=True)
