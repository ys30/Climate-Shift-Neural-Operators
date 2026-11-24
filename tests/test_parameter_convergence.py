"""Parameter totals, complex-FNO accounting and training convergence logging."""
import json
import numpy as np
import pytest
from climate_shift.parameter_audit import inventory
from climate_shift.compare_e1 import compare


def test_exact_baseline_parameter_count():
    result = inventory(width=8, modes=4, grid_size=17)
    assert result["fno"]["torch_parameter_elements"] == 4353
    assert result["fno"]["real_scalar_degrees_of_freedom"] == 8449
    assert result["unet"]["torch_parameter_elements"] == 6753
    assert result["unet"]["real_scalar_degrees_of_freedom"] == 6753
    assert result["persistence"]["torch_parameter_elements"] == 0
    assert result["fno"]["parameter_groups"]["spectral"]["torch_elements"] == 4096
    assert result["fno"]["parameter_groups"]["spectral"]["real_scalar_dof"] == 8192


def test_bad_modes_or_grid_rejected():
    with pytest.raises(ValueError):
        inventory(width=8, modes=10, grid_size=17)
    with pytest.raises(ValueError):
        inventory(width=0, modes=4, grid_size=17)


def test_epoch_history_reports_best_validation_checkpoint(tmp_path):
    rng = np.random.default_rng(14)
    x = (280 + rng.normal(0, 3, size=(12, 1, 8, 8))).astype("float32")
    y = x + rng.normal(0, 0.6, size=x.shape).astype("float32")
    inp = tmp_path / "input.npz"
    out = tmp_path / "report.json"
    np.savez_compressed(inp, X=x, y=y, years=np.repeat([2000, 2001, 2002], 4))
    report = compare(inp, out, 2000, 2001, 2002, epochs=3,
                     batch_size=2, seed=9, width=4, modes=2)
    for name in ("fno", "unet"):
        diag = report["diagnostics"][name]
        assert len(diag["history"]) == 3
        assert [v["epoch"] for v in diag["history"]] == [1, 2, 3]
        assert 1 <= diag["best_epoch"] <= 3
        assert diag["best_val_mse_standardized"] == pytest.approx(
            min(item["validation_mse_standardized"] for item in diag["history"]))
        assert diag["parameter_accounting"]["torch_parameter_elements"] == diag["parameter_count"]
    assert len(json.loads(out.read_text())["diagnostics"]["fno"]["history"]) == 3
