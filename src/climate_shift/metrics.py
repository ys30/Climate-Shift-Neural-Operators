"""Small deterministic metrics for prototype evaluation (not climate-science validation)."""
from __future__ import annotations

import numpy as np


def _paired(pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, r = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    if p.shape != r.shape or p.size == 0 or not (np.all(np.isfinite(p)) and np.all(np.isfinite(r))):
        raise ValueError("Predictions/references must have matching, nonempty, finite shapes")
    return p, r


def regression_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Unweighted toy-grid diagnostics; area weighting is required for full climate analysis."""
    p, r = _paired(pred, ref)
    error = p - r
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(np.mean(error))}


def exceedance_frequency(field: np.ndarray, threshold: float) -> float:
    """Fraction strictly above fixed training-reference threshold (no refit)."""
    x = np.asarray(field, dtype=float)
    if x.size == 0 or not np.all(np.isfinite(x)) or not np.isfinite(threshold):
        raise ValueError("Field and threshold must be finite; field nonempty")
    return float(np.mean(x > threshold))


def warming_response_error(pred_historical: np.ndarray, pred_future: np.ndarray, ref_historical: np.ndarray, ref_future: np.ndarray) -> float:
    """Absolute error of future-minus-historical mean change, in input units.

    Input units and physically comparable time windows are caller responsibilities.
    """
    ph, rh = _paired(pred_historical, ref_historical)
    pf, rf = _paired(pred_future, ref_future)
    return float(abs((pf.mean() - ph.mean()) - (rf.mean() - rh.mean())))


def latitude_weighted_metrics(pred: np.ndarray, ref: np.ndarray,
                              latitude: np.ndarray) -> dict[str, float]:
    """Area-weighted grid metrics for regular lon/lat grids.

    Each longitude cell is assigned a weight proportional to cos(latitude).
    Assumes identical cell area in longitude and time; not a substitute for
    a land-mask or exact grid-cell area when available.
    """
    p, r = _paired(pred, ref)
    if p.ndim != 4 or p.shape[1] != 1:
        raise ValueError("Expect [time, 1, lat, lon] fields")
    lat = np.asarray(latitude, dtype=float)
    if lat.shape != (p.shape[-2],) or not np.isfinite(lat).all() or np.any(np.abs(lat) > 90):
        raise ValueError("Latitude must match grid rows and lie in [-90,90]")
    weights = np.cos(np.deg2rad(lat))
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Latitude weights must sum to a positive value")
    err = p - r
    w = weights.reshape(1, 1, -1, 1)
    total = float(weights.sum() * p.shape[0] * p.shape[1] * p.shape[-1])
    return {
        "mae": float((np.abs(err) * w).sum() / total),
        "rmse": float(np.sqrt(((err ** 2) * w).sum() / total)),
        "bias": float((err * w).sum() / total),
    }
