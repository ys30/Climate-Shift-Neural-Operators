import hashlib
import json
import numpy as np
import pytest
from climate_shift.coverage_gate import check_coverage


def make_fixture(tmp_path, selection="uniform_across_month_valid_pairs", skip=None,
                 bad_lead=False):
    path = tmp_path / "dataset.npz"
    times = np.array(
        [np.datetime64(f"{year}-{month:02d}-{day:02d}T00", "ns")
         for year in (2008, 2009, 2010, 2011, 2012)
         for month in range(1, 13)
         for day in (2, 3, 4)],
        dtype="datetime64[ns]")
    sources = times - np.timedelta64(24, "h")
    if bad_lead:
        sources[0] = times[0] - np.timedelta64(18, "h")
    years = times.astype("datetime64[Y]").astype(int) + 1970
    np.savez_compressed(path, X=np.ones((len(times), 1, 8, 8), dtype="float32"),
                        y=np.ones((len(times), 1, 8, 8), dtype="float32"),
                        years=years, target_times=times, source_times=sources)
    metadata = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_samples": len(times), "target_first": str(times[0]),
        "target_last": str(times[-1]), "forecast_horizon_hours": 24,
        "variable": "2m_temperature", "unit": "K",
        "selection": selection, "requested_years": [2008, 2009, 2010, 2011, 2012],
        "actual_years": [2008, 2009, 2010, 2011, 2012],
        "skipped_months": [] if skip is None else [skip],
    }
    card = tmp_path / "dataset.json"
    card.write_text(json.dumps(metadata))
    return path, card


def test_coverage_passes_full_month_plan(tmp_path):
    path, card = make_fixture(tmp_path)
    result = check_coverage(path, card, [2008, 2009, 2010, 2011, 2012], 3)
    assert result["required_months"] == 60
    assert result["total_samples"] == 180


def test_missing_month_is_rejected(tmp_path):
    path, card = make_fixture(tmp_path, skip="2010-06")
    with pytest.raises(ValueError, match="skipped"):
        check_coverage(path, card, [2008, 2009, 2010, 2011, 2012], 3)


def test_insufficient_month_is_rejected(tmp_path):
    path, card = make_fixture(tmp_path)
    with pytest.raises(ValueError, match="Insufficient"):
        check_coverage(path, card, [2008, 2009, 2010, 2011, 2012], 4)


def test_legacy_sampling_method_is_rejected(tmp_path):
    path, card = make_fixture(tmp_path, selection="first")
    with pytest.raises(ValueError, match="uniform"):
        check_coverage(path, card, [2008, 2009, 2010, 2011, 2012], 3)


def test_bad_lead_is_rejected(tmp_path):
    path, card = make_fixture(tmp_path, bad_lead=True)
    with pytest.raises(ValueError, match="exact"):
        check_coverage(path, card, [2008, 2009, 2010, 2011, 2012], 3)
