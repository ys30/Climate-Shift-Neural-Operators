import numpy as np
import pytest
from scripts.run_e1_multiseed import run_seeds


def test_multiseed(tmp_path):
    rng = np.random.default_rng(15)
    x = rng.normal(280, 3, size=(9, 1, 8, 8)).astype("float32")
    y = x + rng.normal(0, .5, size=x.shape).astype("float32")
    years = np.repeat([2000, 2001, 2002], 3)
    path = tmp_path / "data.npz"
    np.savez_compressed(path, X=x, y=y, years=years)
    summary = run_seeds(path, tmp_path / "runs", 2000, 2001, 2002,
                        seeds=(1, 2), epochs=1, width=4, modes=2)
    assert len(summary["models"]["fno"]["rmse"]["individual"]) == 2
    assert (tmp_path / "runs" / "seed_summary.json").exists()
    with pytest.raises(ValueError):
        run_seeds(path, tmp_path / "dup", 2000, 2001, 2002,
                  seeds=(1, 1), epochs=1, width=4, modes=2)


def test_multiseed_strict_time_provenance(tmp_path):
    rng = np.random.default_rng(20)
    dates = np.array(["2000-02-02T00", "2000-02-04T00",
                      "2001-02-02T00", "2001-02-04T00",
                      "2002-02-02T00", "2002-02-04T00"], dtype="datetime64[ns]")
    origins = dates - np.timedelta64(24, "h")
    x = rng.normal(280, 4, size=(6, 1, 8, 8)).astype("float32")
    y = x + rng.normal(0, 1, size=x.shape).astype("float32")
    path = tmp_path / "dated.npz"
    np.savez_compressed(path, X=x, y=y,
                        years=np.repeat([2000, 2001, 2002], 2),
                        source_times=origins, target_times=dates)
    summary = run_seeds(path, tmp_path / "out", 2000, 2001, 2002,
                        seeds=(11, 22), epochs=1, width=4, modes=2,
                        require_pair_times=True)
    assert summary["split_isolation"]["status"] == "passed"
    assert len(summary["models"]["fno"]["heat_recall"]["individual"]) == 2
    assert len(summary["paired_differences"]["unet_minus_persistence"]["rmse"]["individual"]) == 2


def test_multiseed_weighted_pair_differences(tmp_path):
    rng = np.random.default_rng(61)
    x = rng.normal(290, 4, size=(9, 1, 8, 8)).astype("float32")
    y = x + rng.normal(0, 0.5, size=x.shape).astype("float32")
    path = tmp_path / "weighted.npz"
    np.savez_compressed(path, X=x, y=y, years=np.repeat([2000, 2001, 2002], 3),
                        latitude=np.linspace(25, 50, 8))
    summary = run_seeds(path, tmp_path / "weighted_output", 2000, 2001, 2002,
                        seeds=(10, 20), epochs=1, width=4, modes=2)
    assert summary["area_weighting_status"].startswith("cos(latitude)")
    assert len(summary["paired_differences"]["fno_minus_unet"]["weighted_rmse"]["individual"]) == 2
    assert "weighted_rmse" in summary["models"]["unet"]
