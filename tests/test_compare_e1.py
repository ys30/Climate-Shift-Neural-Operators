import json
import numpy as np
import pytest
import torch
from climate_shift.unet import UNet2d
from climate_shift.compare_e1 import compare

def test_unet_odd_grid_and_gradients():
    net = UNet2d(width=4)
    x = torch.randn(2, 1, 9, 11)
    y = net(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert any(p.grad is not None for p in net.parameters())

def test_three_way_comparison(tmp_path):
    rng = np.random.default_rng(45)
    x = (280 + rng.normal(size=(12, 1, 8, 8))).astype("float32")
    y = (x + rng.normal(scale=.1, size=x.shape)).astype("float32")
    years = np.repeat([2000, 2001, 2002], 4)
    path = tmp_path / "weather.npz"
    np.savez_compressed(path, X=x, y=y, years=years)
    dest = tmp_path / "result.json"
    out = compare(path, dest, 2000, 2001, 2002, epochs=1, width=4, modes=2)
    assert set(out["metrics"]) == {"fno", "unet", "persistence"}
    assert out["n_samples"] == {"train": 4, "validation": 4, "test": 4}
    assert json.loads(dest.read_text())["input_sha256"] == out["input_sha256"]
    for result in out["metrics"].values():
        assert np.isfinite(result["rmse"])
    with pytest.raises(FileExistsError):
        compare(path, dest, 2000, 2001, 2002, epochs=1, width=4, modes=2)
