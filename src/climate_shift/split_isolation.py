"""Cross-partition source/target-time isolation checks for E1.

Years are assigned by target timestamps. Before training, require that all
earlier-fold target times strictly precede every later-fold forecast origin.
This conservative guard prevents sharing boundary instants across folds.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

def audit_split_isolation(path: str | Path, masks: tuple[np.ndarray, np.ndarray, np.ndarray],
                          require_pair_times: bool = False) -> dict:
    with np.load(path, allow_pickle=False) as d:
        keys = set(d.files)
        if not {"source_times", "target_times"} <= keys:
            if require_pair_times:
                raise ValueError("Require both source_times and target_times for a time-isolated run")
            return {"status": "unverified_legacy_no_pair_times"}
        source = d["source_times"].astype("datetime64[ns]")
        target = d["target_times"].astype("datetime64[ns]")
    if source.ndim != 1 or target.shape != source.shape:
        raise ValueError("Malformed source/target times")
    if np.isnat(source).any() or np.isnat(target).any():
        raise ValueError("NaT timestamps are forbidden")
    if not np.all(target - source == np.timedelta64(24, "h")):
        raise ValueError("Not all sample leads are exactly 24 hours")
    if len(masks) != 3 or any(np.asarray(m).shape != source.shape for m in masks):
        raise ValueError("Exactly three masks with matching sample dimension required")
    masks = tuple(np.asarray(m, dtype=bool) for m in masks)
    if any(not m.any() for m in masks):
        raise ValueError("Empty train/validation/test fold")
    if np.any(np.asarray(masks, dtype=int).sum(axis=0) > 1):
        raise ValueError("Overlapping fold membership")
    names = ("train", "validation", "test")
    boundaries = []
    for previous, following, left, right in ((masks[0], masks[1], names[0], names[1]),
                                             (masks[1], masks[2], names[1], names[2]),
                                             (masks[0], masks[2], names[0], names[2])):
        last_old_target = target[previous].max()
        first_new_origin = source[following].min()
        if not last_old_target < first_new_origin:
            raise ValueError(
                f"Boundary contamination: {left} latest target {last_old_target} "
                f"must precede {right} first forecast origin {first_new_origin}"
            )
        boundaries.append({"earlier": left, "later": right,
                           "last_prior_target": str(last_old_target),
                           "first_next_origin": str(first_new_origin),
                           "gap_hours": float((first_new_origin - last_old_target) / np.timedelta64(1, "h"))})
    return {"status": "passed", "boundaries": boundaries,
            "requirement": "strictly earlier target than next fold forecast origin"}
