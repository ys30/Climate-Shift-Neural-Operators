"""Training-climatology-referenced historical ERA5 distribution diagnostics.

This is a descriptive year-to-year shift audit, NOT attribution to climate change.
All climatological references are fit on training targets only, with no validation
or heldout targets used in their construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from climate_shift.data import load_npz_pairs


def audit_historical_shift(path: str | Path, train_end: int = 2010,
                           test_start: int = 2012, test_end: int = 2014) -> dict:
    path = Path(path)
    x, y, years = load_npz_pairs(path)
    with np.load(path, allow_pickle=False) as raw:
        if "target_times" not in raw.files or "latitude" not in raw.files:
            raise ValueError("Need actual target_times and latitude for shift audit")
        times, latitude = raw["target_times"], raw["latitude"]
    if y.shape[1] != 1 or not (train_end < test_start <= test_end):
        raise ValueError("Expected single-temperature channel and valid years")
    months = times.astype("datetime64[M]").astype("int64") % 12 + 1
    train = years <= train_end
    if train.sum() == 0 or not np.any((years >= test_start) & (years <= test_end)):
        raise ValueError("Training or test cohort empty")
    # Training-only local monthly temperature climatology.
    mu, std, counts = [], [], []
    for month in range(1, 13):
        subset = y[train & (months == month), 0].astype(float)
        if len(subset) < 20:
            raise ValueError(f"Month {month} has fewer than 20 training target samples")
        mu.append(subset.mean(axis=0))
        std.append(subset.std(axis=0, ddof=1))
        counts.append(int(len(subset)))
    mu = np.stack(mu)
    std = np.stack(std)
    # Robust scale avoids division by zero without deriving a reference from test data.
    if (not np.isfinite(std).all()) or np.any(std < 1e-6):
        raise ValueError("Unstable training-only month-local temperature scale")
    anomaly = y[:, 0].astype(float) - mu[months - 1]
    z = anomaly / std[months - 1]
    w = np.cos(np.deg2rad(latitude.astype(float)))
    if np.any(w <= 0):
        raise ValueError("Positive area weights required")
    weights = np.broadcast_to(w.reshape(1, -1, 1), anomaly.shape)
    output = {}
    for year in sorted(np.unique(years)):
        if not (year <= train_end or test_start <= year <= test_end):
            continue
        selected = years == year
        ay, zy, wy = anomaly[selected], z[selected], weights[selected]
        n = int(selected.sum())
        area_mass = float(wy.sum())
        output[str(int(year))] = {
            "n_pairs": n,
            "mean_anomaly_K": float((ay * wy).sum() / area_mass),
            "mean_standardized_anomaly": float((zy * wy).sum() / area_mass),
            "fraction_gridcell_observations_z_gt_2": float(
                (wy * (zy > 2)).sum() / area_mass
            ),
            "fraction_gridcell_observations_z_lt_minus_2": float(
                (wy * (zy < -2)).sum() / area_mass
            ),
            "region_mean_anomaly_K_by_sample_mean": float(
                (ay * wy).sum(axis=(1, 2)).mean() / float(w.sum() * y.shape[-1])
            ),
        }
    train_anomaly = anomaly[train]
    train_w = weights[train]
    train_reference_mean = float((train_anomaly * train_w).sum() / train_w.sum())
    return {
        "status": "historical_descriptive_shift_not_climate_change_attribution",
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_end": train_end, "test_start": test_start, "test_end": test_end,
        "train_counts_by_month": {f"{i+1:02d}": n for i, n in enumerate(counts)},
        "train_only_month_local_climatology": True,
        "training_weighted_reference_anomaly_K": train_reference_mean,
        "per_year": output,
        "limitations": [
            "Sparse 24-pairs/month cohort: not complete annual daily or hourly climatology",
            "Anomaly means and exceedance fractions summarize selected gridcell samples, not independent events",
            "No causal warming attribution or future OOD climate response from historical anomalies",
            "Shifts are descriptive; independent physical baselines and CMIP6 must be evaluated separately",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-end", type=int, default=2010)
    ap.add_argument("--test-start", type=int, default=2012)
    ap.add_argument("--test-end", type=int, default=2014)
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = audit_historical_shift(args.input, args.train_end, args.test_start, args.test_end)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
