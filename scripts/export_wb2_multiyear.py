"""Bounded multi-year ERA5/WB2 development extract, stratified across months.

Requires optional ERA5 dependencies and access to official remote Zarr.
This is a small development extract, NOT a finalized climate validation corpus.
"""
from __future__ import annotations
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import numpy as np
import xarray as xr
from climate_shift.era5_adapter import make_24h_pairs
from climate_shift.data import load_npz_pairs
DEFAULT_STORE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

def extract(ds, years, max_per_month=8, stride=4):
    if max_per_month < 1 or stride < 1:
        raise ValueError("max_per_month and stride must be positive")
    years = [int(year) for year in years]
    if len(set(years)) != len(years) or years != sorted(years):
        raise ValueError("Years must be unique and in increasing order")
    x_all, y_all, yrs_all, times_all = [], [], [], []
    skipped = []
    for year in years:
        for month in range(1, 13):
            start = date(year, month, 1)
            stop = date(year + (month == 12), (month % 12) + 1, 1)
            end = date.fromordinal(stop.toordinal() - 1)
            try:
                x, y, yy, stamps = make_24h_pairs(
                    ds, start=start.isoformat(), end=end.isoformat(),
                    stride=stride, max_samples=max_per_month)
            except ValueError as exc:
                # Only skip months genuinely outside the published data window.
                if "No timestamps inside requested date window" not in str(exc) or "No exact 24-hour pairs" in str(exc):
                    raise
                skipped.append(start.isoformat())
                continue
            if not np.all(yy == year):
                raise ValueError("Target timestamps escaped their source year")
            x_all.append(x)
            y_all.append(y)
            yrs_all.append(yy)
            times_all.append(stamps)
    if not x_all:
        raise ValueError("No samples across requested years")
    target_times = np.concatenate(times_all)
    if len(np.unique(target_times)) != len(target_times) or not np.all(target_times[1:] > target_times[:-1]):
        raise ValueError("Targets duplicated or not chronologically ordered")
    return (np.concatenate(x_all), np.concatenate(y_all),
            np.concatenate(yrs_all), target_times, skipped)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--years", type=int, nargs="+", required=True)
    p.add_argument("--max-per-month", type=int, default=8)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix(".json").exists():
        p.error("Refusing to overwrite an existing dataset or metadata")
    ds = xr.open_zarr(args.store, consolidated=True, chunks=None)
    try:
        x, y, years, times, skipped = extract(ds, args.years, args.max_per_month, args.stride)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, X=x, y=y, years=years, target_times=times)
        # Verify output round-trip and fail on schema corruption.
        check_x, check_y, check_years = load_npz_pairs(out)
        if check_x.shape != x.shape or check_y.shape != y.shape or not np.array_equal(check_years, years):
            raise RuntimeError("Post-export validation failed")
        meta = {"source": args.store, "requested_years": args.years,
                "actual_years": np.unique(years).tolist(), "max_per_month": args.max_per_month,
                "stride": args.stride, "n_samples": len(years), "shape": list(x.shape),
                "target_first": str(times[0]), "target_last": str(times[-1]),
                "skipped_months": skipped, "forecast_horizon_hours": 24,
                "unit": ds["2m_temperature"].attrs.get("units", "UNVERIFIED"),
                "variable": "2m_temperature",
                "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "status": "bounded development data, NOT full ERA5 evaluation"}
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2))
    finally:
        ds.close()

if __name__ == "__main__":
    main()
