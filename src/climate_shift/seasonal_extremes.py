"""Training-only monthly and grid-local warm-tail diagnostics.

These are discrete hot grid-cell exceedances, not heatwave events. A monthly
threshold estimated on sparse sampling is exploratory and not a robust
daily-climatology heatwave threshold.
"""
from __future__ import annotations

import numpy as np


def _months(times: np.ndarray, n_expected: int) -> np.ndarray:
    times = np.asarray(times)
    if times.shape != (n_expected,) or times.dtype.kind != "M" or np.isnat(times).any():
        raise ValueError("Expected valid datetime64 timestamps aligned to samples")
    return (times.astype("datetime64[M]").astype("int64") % 12 + 1).astype(int)


def fit_local_monthly_p95(training: np.ndarray, train_times: np.ndarray,
                          quantile: float = .95, min_samples_per_month: int = 20
                          ) -> tuple[np.ndarray, dict]:
    """Returns thresholds [12,H,W] from training targets only."""
    values = np.asarray(training, dtype=float)
    if (values.ndim != 4 or values.shape[1] != 1 or not values.size or
            not np.isfinite(values).all()):
        raise ValueError("Training fields must be finite [N,1,H,W]")
    if not (0 < quantile < 1) or min_samples_per_month < 2:
        raise ValueError("Require 0<quantile<1 and at least 2 samples per month")
    months = _months(train_times, len(values))
    q = np.empty((12, values.shape[-2], values.shape[-1]), dtype=float)
    counts = {}
    for m in range(1, 13):
        monthly = values[months == m, 0]
        counts[f"{m:02d}"] = int(len(monthly))
        if len(monthly) < min_samples_per_month:
            raise ValueError(
                f"Insufficient training samples for month {m}: {len(monthly)} "
                f"(require {min_samples_per_month})"
            )
        q[m - 1] = np.quantile(monthly, quantile, axis=0)
    return q, {
        "quantile": float(quantile),
        "min_samples_per_month": int(min_samples_per_month),
        "actual_samples_per_month": counts,
        "threshold_min_K": float(q.min()),
        "threshold_max_K": float(q.max()),
        "fitting_scope": "training_targets_only_monthly_local_gridcell",
        "interpretation": "exploratory exceedance diagnostic, not a heatwave event",
    }


def local_monthly_hot_diagnostics(pred: np.ndarray, ref: np.ndarray,
                                  target_times: np.ndarray, thresholds: np.ndarray,
                                  latitude: np.ndarray | None = None) -> dict:
    p, r = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    if (p.shape != r.shape or p.ndim != 4 or p.shape[1] != 1 or not p.size or
            not np.isfinite(p).all() or not np.isfinite(r).all()):
        raise ValueError("Expected finite paired [N,1,H,W] fields")
    q = np.asarray(thresholds, dtype=float)
    if q.shape != (12, r.shape[-2], r.shape[-1]) or not np.isfinite(q).all():
        raise ValueError("Threshold array must be finite [12,H,W] matching field grid")
    months = _months(target_times, len(r))
    cut = q[months - 1, None, :, :]
    if latitude is None:
        weight_lat = np.ones(r.shape[-2], dtype=float)
        weight_status = "unweighted"
    else:
        lat = np.asarray(latitude, dtype=float)
        if lat.shape != (r.shape[-2],) or not np.isfinite(lat).all() or np.any(np.abs(lat) > 90):
            raise ValueError("Latitude coordinates must match grid rows")
        weight_lat = np.cos(np.deg2rad(lat))
        weight_status = "cos(latitude)"
    w = np.broadcast_to(weight_lat.reshape(1, 1, -1, 1), r.shape)
    true_hot, pred_hot = r > cut, p > cut
    true_mass = float((w * true_hot).sum())
    pred_mass = float((w * pred_hot).sum())
    hit_mass = float((w * (true_hot & pred_hot)).sum())
    total = float(w.sum())
    if total <= 0:
        raise ValueError("No positive grid weights")
    def finite_or_none(x):
        return float(x) if np.isfinite(x) else None
    return {
        "reference_hot_fraction": true_mass / total,
        "predicted_hot_fraction": pred_mass / total,
        "frequency_bias": (pred_mass - true_mass) / total,
        "hot_recall": finite_or_none(hit_mass / true_mass) if true_mass else None,
        "hot_precision": finite_or_none(hit_mass / pred_mass) if pred_mass else None,
        "hot_conditional_bias": finite_or_none(float((w * (p-r) * true_hot).sum() / true_mass))
                                if true_mass else None,
        "hot_reference_count": int(true_hot.sum()),
        "weighting": weight_status,
        "threshold_definition": "train_only_monthly_gridcell_p95",
    }
