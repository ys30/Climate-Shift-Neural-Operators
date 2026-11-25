import numpy as np
import pytest
from climate_shift.compare_e1 import compare
from climate_shift.parameter_audit import inventory


def test_independent_unet_width_matches_declared_parameter_near_pair(tmp_path):
    rng = np.random.default_rng(102)
    x = (290 + rng.normal(0, 2, (12, 1, 17, 17))).astype("float32")
    y = (x + rng.normal(0, .4, x.shape)).astype("float32")
    src = tmp_path / "data.npz"
    np.savez_compressed(src, X=x, y=y, years=np.repeat([2008, 2011, 2012], 4))
    result = compare(src, tmp_path / "out.json", 2008, 2011, 2012,
                     epochs=1, width=8, modes=4, unet_width=9)
    assert result["hyperparameters"]["width"] == 8
    assert result["hyperparameters"]["unet_width"] == 9
    assert result["diagnostics"]["fno"]["parameter_accounting"]["real_scalar_degrees_of_freedom"] == (
        inventory(width=8, modes=4)["fno"]["real_scalar_degrees_of_freedom"]
    )
    assert result["diagnostics"]["unet"]["parameter_accounting"]["real_scalar_degrees_of_freedom"] == (
        inventory(width=9, modes=1)["unet"]["real_scalar_degrees_of_freedom"]
    )


def test_reject_invalid_independent_width(tmp_path):
    src = tmp_path / "input.npz"
    np.savez_compressed(src, X=np.ones((3, 1, 17, 17)), y=np.ones((3, 1, 17, 17)),
                        years=np.array([2008, 2011, 2012]))
    with pytest.raises(ValueError, match="Invalid training hyperparameters"):
        compare(src, tmp_path / "bad.json", 2008, 2011, 2012,
                epochs=1, width=8, modes=4, unet_width=0)
