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
        if set(data.files) not in ({"X", "y", "years"}, {"X", "y", "years", "target_times"}):
            raise ValueError("NPZ must contain X, y, years and optional target_times")
        x, y, years = data["X"], data["y"], data["years"]
        target_times = data["target_times"] if "target_times" in data.files else None
    if x.ndim != 4 or y.ndim != 4 or x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[2:]:
        raise ValueError("Expected matched [N,C,H,W] grid arrays")
    if len(years) != len(x) or years.ndim != 1 or years.dtype.kind not in 'iu':
        raise ValueError("years must be integer [N] target timestamp years")
    if target_times is not None:
        if (target_times.shape != (len(years),) or
                target_times.dtype.kind != "M" or np.isnat(target_times).any()):
            raise ValueError("target_times must be valid datetime64 vector")
        if len(target_times) > 1 and not np.all(np.diff(target_times) > np.timedelta64(0, "ns")):
            raise ValueError("target_times must be unique and strictly increasing")
        computed_years = target_times.astype("datetime64[Y]").astype(int) + 1970
        if not np.array_equal(computed_years, years):
            raise ValueError("target_times and years disagree")
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
