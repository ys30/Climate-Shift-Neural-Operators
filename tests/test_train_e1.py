import numpy as np
import pytest
from climate_shift.train_e1 import run

def test_train_eval_pipeline(tmp_path):
    rng = np.random.default_rng(120)
    x = rng.normal(280, 3, (12, 1, 8, 8)).astype("float32")
    y = x + rng.normal(0, 0.1, x.shape).astype("float32")
    years = np.repeat([2000, 2001, 2002], 4)
    data = tmp_path / "pairs.npz"
    np.savez_compressed(data, X=x, y=y, years=years)
    output = tmp_path / "metrics.json"
    result = run(data, 2000, 2001, 2002, output, epochs=1, width=4, modes=2)
    assert result["sample_counts"] == {"train": 4, "validation": 4, "test": 4}
    assert output.is_file()
    assert abs(result["normalization"]["target_mean_train_only"] - y[:4].mean()) < 0.0001
    assert np.isfinite(result["test"]["FNO"]["mae"])
    with pytest.raises(FileExistsError):
        run(data, 2000, 2001, 2002, output, epochs=1, width=4, modes=2)

def test_single_year_cannot_be_used_for_three_splits(tmp_path):
    data = tmp_path / "one_year.npz"
    np.savez_compressed(data, X=np.ones((4,1,8,8)), y=np.ones((4,1,8,8)), years=np.repeat(2019, 4))
    with pytest.raises(ValueError):
        run(data, 2017, 2018, 2019, tmp_path / "out.json", epochs=1, width=4, modes=2)
