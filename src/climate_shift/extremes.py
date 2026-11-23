"""Train-referenced extreme-temperature diagnostics for gridded fields.

A single training-derived scalar threshold is deliberately a development
diagnostic; a scientific heatwave definition needs seasonality and duration.
"""
from __future__ import annotations
import numpy as np


def fit_hot_threshold(training: np.ndarray, quantile: float = 0.95) -> float:
    x = np.asarray(training, dtype=float)
    if x.ndim != 4 or x.shape[1] != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Require finite [time,1,lat,lon] training reference")
    if not 0 < quantile < 1:
        raise ValueError("Quantile must lie strictly between zero and one")
    return float(np.quantile(x, quantile))


def hot_diagnostics(pred: np.ndarray, ref: np.ndarray, threshold: float,
                    latitude: np.ndarray | None = None) -> dict[str, float | int]:
    """Tail bias, recall, precision and exceedance-frequency bias.

    Optional cos(latitude) area weighting; these are grid exceedances,
    NOT independent heatwave events.
    """
    p, r = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    if p.shape != r.shape or p.ndim != 4 or p.shape[1] != 1 or p.size == 0:
        raise ValueError("Expected matching nonempty [time,1,lat,lon] arrays")
    if not (np.isfinite(p).all() and np.isfinite(r).all() and np.isfinite(threshold)):
        raise ValueError("All inputs must be finite")
    if latitude is None:
        ww = np.ones_like(r)
    else:
        lat = np.asarray(latitude, dtype=float)
        if lat.shape != (r.shape[-2],) or not np.isfinite(lat).all() or np.any(np.abs(lat) > 90):
            raise ValueError("Latitude must be one valid degree value per grid row")
        ww = np.broadcast_to(np.cos(np.deg2rad(lat)).reshape(1, 1, -1, 1), r.shape)
    total = float(ww.sum())
    if total <= 0:
        raise ValueError("No positive area weights")
    hot_true, hot_pred = r > threshold, p > threshold
    denom_true = float((ww * hot_true).sum())
    denom_pred = float((ww * hot_pred).sum())
    hits = float((ww * (hot_true & hot_pred)).sum())
    return {
        "threshold": float(threshold),
        "reference_hot_fraction": denom_true / total,
        "predicted_hot_fraction": denom_pred / total,
        "frequency_bias": (denom_pred - denom_true) / total,
        "hot_conditional_bias": float((ww * (p - r) * hot_true).sum() / denom_true) if denom_true else float("nan"),
        "hot_recall": hits / denom_true if denom_true else float("nan"),
        "hot_precision": hits / denom_pred if denom_pred else float("nan"),
        "hot_reference_count": int(hot_true.sum()),
    }
