"""Split validation and training-only standardization.

This package deliberately does not infer a split from shuffled samples.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np


def validate_chronological_splits(train_years: Sequence[int], val_years: Sequence[int], test_years: Sequence[int], embargo_years: int = 0) -> None:
    """Reject temporal overlap, reversed splits, and insufficient year embargo."""
    if embargo_years < 0:
        raise ValueError("embargo_years must be nonnegative")
    arrays = [np.asarray(a, dtype=int) for a in (train_years, val_years, test_years)]
    if any(a.ndim != 1 or not len(a) for a in arrays):
        raise ValueError("Each year split must be a nonempty 1-D sequence")
    if any(len(np.unique(a)) != len(a) for a in arrays):
        raise ValueError("Duplicate years within a split")
    for earlier, later in zip(arrays, arrays[1:]):
        if earlier.max() + embargo_years >= later.min():
            raise ValueError("Chronological splits overlap, are reversed, or violate embargo")


def train_only_standardize(train: np.ndarray, other: np.ndarray, axis: int | tuple[int, ...] = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit mean/std exclusively on training data; apply to both sets.

    Other is never used to compute normalization statistics. NaNs are rejected.
    """
    train = np.asarray(train, dtype=float)
    other = np.asarray(other, dtype=float)
    if not (np.all(np.isfinite(train)) and np.all(np.isfinite(other))):
        raise ValueError("Non-finite input; apply explicit missing-data policy first")
    mean = train.mean(axis=axis, keepdims=True)
    std = train.std(axis=axis, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (train - mean) / std, (other - mean) / std, mean, std
