"""Strict NPZ-based E1 data contract: no download, no preprocessing guesses."""
from __future__ import annotations
from pathlib import Path
import numpy as np

def load_npz_pairs(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y of shape [time,channels,lat,lon], years of shape [time].

    X and y must already be in the same physical units and grid. Years refer to
    target timestamps; upstream must prove predictor/target time alignment.
    """
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"X", "y", "years"}:
            raise ValueError("NPZ must contain exactly X, y, years")
        x, y, years = data["X"], data["y"], data["years"]
    if x.ndim != 4 or y.ndim != 4 or x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[2:]:
        raise ValueError("Expected matched [N,C,H,W] grid arrays")
    if len(years) != len(x) or years.ndim != 1 or years.dtype.kind not in 'iu':
        raise ValueError("years must be integer [N] target timestamp years")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("NaN/Inf found; decide missing-data treatment upstream")
    return x.astype(np.float32), y.astype(np.float32), years.astype(int)

def chronological_masks(years: np.ndarray, train_end: int, val_end: int, test_end: int, embargo_years: int = 0):
    """Return masks with explicit full-year embargo at each boundary."""
    years = np.asarray(years)
    if embargo_years < 0 or not (train_end < val_end < test_end):
        raise ValueError("Invalid boundaries or embargo")
    masks = (years <= train_end,
             (years > train_end + embargo_years) & (years <= val_end),
             (years > val_end + embargo_years) & (years <= test_end))
    if any(not np.any(m) for m in masks):
        raise ValueError("At least one chronological partition is empty")
    if np.any(years > test_end):
        raise ValueError("Dataset contains years beyond declared test end")
    return masks
