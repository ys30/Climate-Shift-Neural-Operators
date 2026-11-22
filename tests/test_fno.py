import numpy as np
import pytest
import torch
from climate_shift.models import SpectralConv2d, FNO2d
from climate_shift.data import chronological_masks, load_npz_pairs

def test_fno_output_and_gradient():
    model = FNO2d(width=4, modes_y=2, modes_x=2, depth=2)
    x = torch.randn(2, 1, 8, 8)
    y = model(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_spectral_grid_validation():
    with pytest.raises(ValueError):
        SpectralConv2d(1, 1, 5, 5)(torch.randn(2, 1, 8, 8))

def test_npz_contract(tmp_path):
    f = tmp_path / 'sample.npz'
    np.savez(f, X=np.ones((3, 1, 8, 8)), y=np.ones((3, 1, 8, 8)), years=np.array([2000, 2003, 2006]))
    x, y, yrs = load_npz_pairs(f)
    assert x.shape == (3, 1, 8, 8)
    masks = chronological_masks(yrs, train_end=2000, val_end=2003, test_end=2006, embargo_years=1)
    assert [int(m.sum()) for m in masks] == [1, 1, 1]
    with pytest.raises(ValueError):
        chronological_masks(yrs, train_end=2001, val_end=2003, test_end=2006, embargo_years=2)
