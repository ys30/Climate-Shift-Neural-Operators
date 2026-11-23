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
