import hashlib
import json
import numpy as np
import pytest
from climate_shift.audit_e1 import audit
from climate_shift.data import load_npz_pairs


def fixture(tmp_path):
    path = tmp_path / "sample.npz"
    times = np.array(["2000-01-01T00", "2000-02-01T00", "2001-01-01T00"], dtype="datetime64[ns]")
    np.savez_compressed(path, X=np.ones((3,1,8,8)), y=np.ones((3,1,8,8)),
                        years=np.array([2000,2000,2001]), target_times=times)
    meta = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n_samples": 3,
            "target_first": str(times[0]), "target_last": str(times[-1]),
            "forecast_horizon_hours": 24, "variable": "2m_temperature",
            "source": "TEST_SYNTHETIC", "unit": "K"}
    card = tmp_path / "sample.json"
    card.write_text(json.dumps(meta))
    return path, card


def test_audited_timestamps_and_manifest(tmp_path):
    path, card = fixture(tmp_path)
    result = audit(path, card)
    assert result["metadata_verified"] is True
    assert result["years"] == {"2000": 2, "2001": 1}
    assert result["samples_per_month"]["2000-01"] == 1


def test_wrong_checksum_is_rejected(tmp_path):
    path, card = fixture(tmp_path)
    meta = json.loads(card.read_text())
    meta["sha256"] = "incorrect"
    card.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="SHA-256"):
        audit(path, card)


def test_year_mismatch_or_nonmonotonic_timestamp_is_rejected(tmp_path):
    path, _ = fixture(tmp_path)
    with np.load(path) as d:
        x,y,years,times = (d[k] for k in ("X","y","years","target_times"))
    np.savez_compressed(path,X=x,y=y,years=years,target_times=times[::-1])
    with pytest.raises(ValueError):
        load_npz_pairs(path)
    np.savez_compressed(path,X=x,y=y,years=np.array([1999,2000,2001]),target_times=times)
    with pytest.raises(ValueError, match="disagree"):
        load_npz_pairs(path)


def test_legacy_npz_requires_timestamps_for_audit(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path,X=np.ones((2,1,8,8)),y=np.ones((2,1,8,8)),years=[2000,2001])
    load_npz_pairs(path)
    with pytest.raises(ValueError, match="target_times"):
        audit(path)
