"""Train-climatology anomaly-stratified forecast diagnostics.

Use only training target fields to construct local monthly mean/std; never
select thresholds or reference moments from test/validation observations.
"""
from __future__ import annotations

import numpy as np
from .seasonal_extremes import _months


def fit_monthly_standardization(training: np.ndarray, train_times: np.ndarray,
                                min_per_month: int = 20) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(training, dtype=float)
    if y.ndim != 4 or y.shape[1] != 1 or not np.isfinite(y).all():
        raise ValueError("Training data must be finite [N,1,H,W]")
    months = _months(train_times, len(y))
    means, stds = [], []
    for m in range(1, 13):
        vals = y[months == m, 0]
        if len(vals) < min_per_month:
            raise ValueError(f"Insufficient month {m} training observations")
        means.append(vals.mean(axis=0))
        std = vals.std(axis=0, ddof=1)
        if np.any(std < 1e-6) or not np.isfinite(std).all():
            raise ValueError(f"Unstable month {m} training standard deviation")
        stds.append(std)
    return np.stack(means), np.stack(stds)


def conditional_error(pred: np.ndarray, truth: np.ndarray,
                      test_times: np.ndarray, monthly_mean: np.ndarray,
                      monthly_std: np.ndarray, latitude: np.ndarray) -> dict:
    p, y = np.asarray(pred, float), np.asarray(truth, float)
    if p.shape != y.shape or y.ndim != 4 or y.shape[1] != 1 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Require finite, same-shape [N,1,H,W] predictions and truth")
    n, _, h, w = y.shape
    if monthly_mean.shape != (12,h,w) or monthly_std.shape != (12,h,w) or np.any(monthly_std <= 0):
        raise ValueError("Require positive train-only 12-month local-gridcell reference")
    latitude = np.asarray(latitude, float)
    if latitude.shape != (h,) or not np.isfinite(latitude).all() or np.any(np.abs(latitude) >= 90):
        raise ValueError("Latitude vector must match grid")
    months = _months(test_times, n)
    z = (y[:,0] - monthly_mean[months-1]) / monthly_std[months-1]
    err = p[:,0] - y[:,0]
    weights = np.broadcast_to(np.cos(np.deg2rad(latitude)).reshape(1,h,1), z.shape)
    bins = [
        ("cold_lt_m2", z < -2),
        ("cool_m2_m1", (z >= -2) & (z < -1)),
        ("normal_m1_p1", (z >= -1) & (z <= 1)),
        ("warm_p1_p2", (z > 1) & (z <= 2)),
        ("hot_gt_p2", z > 2),
    ]
    out = {}
    all_weight = float(weights.sum())
    for label, mask in bins:
        mass = float((weights * mask).sum())
        out[label] = {
            "weighted_fraction": mass / all_weight,
            "weighted_rmse_K": float(np.sqrt((weights * mask * err**2).sum() / mass)) if mass else None,
            "weighted_mae_K": float((weights * mask * np.abs(err)).sum() / mass) if mass else None,
            "weighted_bias_K": float((weights * mask * err).sum() / mass) if mass else None,
            "unweighted_gridcell_count": int(mask.sum()),
        }
    assert abs(sum(v["weighted_fraction"] for v in out.values()) - 1) < 1e-8
    return {"status":"descriptive_train_only_month_local_standardized_anomaly_bins",
            "reference":"training_target_monthly_gridcell_mean_std",
            "bins":out}
