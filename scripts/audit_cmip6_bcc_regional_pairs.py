"""BCC-CSM2-MR cross-model replication stage: bounded native regional +1-day audit.

Source checks, not training or skill inference. Exactly two predeclared dates
per experiment, two regional fields for each date, with no global tas reads.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

try:
    from .audit_cmip6_awi_daily_pairs import contiguous_span, exact_daily_pair_indices
except ImportError:
    from audit_cmip6_awi_daily_pairs import contiguous_span, exact_daily_pair_indices

DATES = {
    "historical": ((2010, 1, 15), (2010, 7, 15)),
    "ssp245": ((2030, 1, 15), (2050, 7, 15)),
    "ssp585": ((2030, 1, 15), (2050, 7, 15)),
}
IDENTITY = ("BCC-CSM2-MR", "r1i1p1f1", "gn")


def get_model(manifest):
    if manifest.get("status") != "catalog_metadata_only_no_climate_fields_or_transfer_skill":
        raise ValueError("Only verified catalog manifests allowed")
    rows = [x for x in manifest.get("selected_candidates", [])
            if (x.get("source_id"), x.get("member_id"), x.get("grid_label")) == IDENTITY]
    if len(rows) != 1:
        raise ValueError("BCC member-grid must match exactly one audited entry")
    row = rows[0]
    for experiment in DATES:
        source = row[experiment]["store"]
        if (not source.startswith("gs://cmip6/")
                or f"/{experiment}/" not in source
                or "/day/tas/" not in source):
            raise ValueError("Invalid frozen BCC source store")
    return row


def sample_one(store, dates):
    import fsspec
    import numpy as np
    import xarray as xr
    ds = xr.open_zarr(fsspec.get_mapper(store, anon=True, token="anon"),
                      consolidated=True, decode_times=False, chunks=None,
                      mask_and_scale=True)
    try:
        tas = ds["tas"]
        if tas.attrs.get("units") != "K" or set(tas.dims) != {"time", "lat", "lon"}:
            raise ValueError("Expect day/tas Kelvin on rectilinear native lat/lon grid")
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float) % 360
        slat = contiguous_span(lat, 25.0, 50.0)
        slon = contiguous_span(lon, 235.0, 260.0)
        lats = lat[slat]
        lons = lon[slon]
        shape = (len(lats), len(lons))
        if not 16 <= shape[0] * shape[1] <= 2000:
            raise ValueError("Unsafe regional grid size")
        t = ds["time"]
        units = str(t.attrs.get("units", ""))
        calendar = str(t.attrs.get("calendar", t.encoding.get("calendar", "standard")))
        if "since" not in units:
            raise ValueError("Missing CF time units")
        numeric = np.asarray(t.values, dtype=float)
        results = []
        weights = np.cos(np.deg2rad(lats))[:, None]
        for scheduled in dates:
            i, j, first, second = exact_daily_pair_indices(numeric, scheduled, units, calendar)
            x = np.asarray(tas.isel(time=i, lat=slat, lon=slon).values, dtype=float)
            y = np.asarray(tas.isel(time=j, lat=slat, lon=slon).values, dtype=float)
            if x.shape != shape or y.shape != shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Sampled BCC data shape or missingness violation")
            if min(float(x.min()), float(y.min())) < 150 or max(float(x.max()), float(y.max())) > 350:
                raise ValueError("Implausible Kelvin temperature")
            score = float(np.sqrt(np.sum(weights * (y-x)**2) / (weights.sum() * shape[1])))
            results.append({
                "origin_native": first, "target_native": second,
                "n_cells": int(x.size),
                "persistence_weighted_rmse_K_sanity_only": score,
                "source_kelvin_min": float(x.min()), "source_kelvin_max": float(x.max()),
                "target_kelvin_min": float(y.min()), "target_kelvin_max": float(y.max()),
                "finite_values": True,
            })
        return {
            "status": "actual_bcc_native_one_day_pair_sanity_only",
            "store": store, "calendar": calendar, "time_units": units,
            "grid_shape": list(shape),
            "latitude_bounds": [float(lats.min()), float(lats.max())],
            "longitude_east_bounds": [float(lons.min()), float(lons.max())],
            "latitude_sha256": __import__("hashlib").sha256(lats.tobytes()).hexdigest(),
            "longitude_sha256": __import__("hashlib").sha256(lons.tobytes()).hexdigest(),
            "pairs": results, "raw_grids_archived": False,
        }
    finally:
        ds.close()


def audit(manifest):
    row = get_model(manifest)
    out = {name: sample_one(row[name]["store"], dates)
           for name, dates in DATES.items()}
    if len({tuple(x["grid_shape"]) for x in out.values()}) != 1:
        raise ValueError("Historical/SSP spatial grid shapes differ")
    if len({(x["latitude_sha256"], x["longitude_sha256"]) for x in out.values()}) != 1:
        raise ValueError("Historical/SSP actual spatial coordinate values differ")
    if len({x["calendar"] for x in out.values()}) != 1:
        raise ValueError("Historical/SSP native calendars differ")
    return {
        "status": "bcc_second_model_actual_regional_pairs_verified_no_neural_training",
        "source_id": IDENTITY[0], "member_id": IDENTITY[1],
        "grid_label": IDENTITY[2],
        "source_catalog_sha256": manifest["catalog_sha256"],
        "scenario_diagnostics": out,
        "limitations": [
            "Only two sparse date pairs per scenario",
            "Not a trained-model transfer or SSP forecast result",
            "Data are CMIP6 scenario simulations, not independent future observations",
            "Grid/source references must be calibrated on BCC Historical, not AWI",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError("Refuse to overwrite results")
    report = audit(json.loads(Path(a.manifest).read_text()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": report["status"],
        "scenarios": {k: {"shape": v["grid_shape"], "calendar": v["calendar"],
                          "days": [(a["origin_native"], a["target_native"])
                                   for a in v["pairs"]]}
                      for k, v in report["scenario_diagnostics"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
