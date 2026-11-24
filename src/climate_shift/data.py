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
        allowed = {"X", "y", "years", "target_times", "source_times", "latitude"}
        keys = set(data.files)
        if not {"X", "y", "years"} <= keys or keys - allowed:
            raise ValueError("NPZ must contain X, y, years and optional time arrays")
        if "source_times" in keys and "target_times" not in keys:
            raise ValueError("source_times require target_times")
        x, y, years = data["X"], data["y"], data["years"]
        target_times = data["target_times"] if "target_times" in data.files else None
        source_times = data["source_times"] if "source_times" in data.files else None
        latitude = data["latitude"] if "latitude" in data.files else None
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
    if source_times is not None:
        if (source_times.shape != (len(years),) or
                source_times.dtype.kind != "M" or np.isnat(source_times).any()):
            raise ValueError("source_times must be valid datetime64 vector")
        if not np.all(target_times.astype("datetime64[ns]") -
                      source_times.astype("datetime64[ns]") == np.timedelta64(24, "h")):
            raise ValueError("Every source-target pair must have an exact +24h lead")
        if len(source_times) > 1 and not np.all(np.diff(source_times) > np.timedelta64(0, "ns")):
            raise ValueError("source_times must be unique and strictly increasing")
    if latitude is not None:
        if (latitude.shape != (x.shape[-2],) or not np.issubdtype(latitude.dtype, np.number)
                or not np.isfinite(latitude).all() or np.any(np.abs(latitude) > 90)):
            raise ValueError("latitude must be finite degree coordinates matching grid rows")
        if len(np.unique(latitude)) != len(latitude):
            raise ValueError("latitude must have unique grid-row coordinates")
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
