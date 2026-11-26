"""Read-only CMIP6 daily TAS Zarr compatibility check.

Opens only metadata: no full climate-field reads, regridding or model fitting.
Each ensemble must retain exactly matching source/member/grid across experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_store(store: str) -> dict:
    if not store.startswith("gs://cmip6/"):
        raise ValueError("Expected an audited public gs://cmip6/ Zarr store")
    try:
        import fsspec
        import numpy as np
        import xarray as xr
        mapper = fsspec.get_mapper(store, anon=True, token="anon")
        ds = xr.open_zarr(mapper, consolidated=True, decode_times=False,
                          chunks=None, mask_and_scale=False)
        try:
            required = ("tas", "time")
            missing = [x for x in required if x not in ds.variables]
            if missing:
                raise ValueError("Required variables absent: " + ", ".join(missing))
            tas = ds["tas"]
            time = ds["time"]
            if "time" not in tas.dims or tas.ndim != 3:
                raise ValueError("Daily tas requires (time,lat,lon)-like three dimensions")
            if tas.sizes["time"] < 365:
                raise ValueError("Fewer than 365 daily tas observations")
            units = str(tas.attrs.get("units", ""))
            if units not in ("K", "kelvin", "Kelvin"):
                raise ValueError(f"Expected absolute-temperature Kelvin, got {units!r}")
            calendar = str(time.attrs.get("calendar", time.encoding.get("calendar", "standard")))
            allowed_calendars = {
                "standard", "gregorian", "proleptic_gregorian", "365_day", "noleap",
                "360_day", "all_leap", "366_day", "julian",
            }
            if calendar not in allowed_calendars:
                raise ValueError(f"Unsupported/unrecognized calendar: {calendar}")
            if not all(k in ds.variables for k in ("lat", "lon")):
                raise ValueError("Required geographic coordinates lat/lon missing")
            lat, lon = ds["lat"], ds["lon"]
            if not (lat.ndim in (1,2) and lon.ndim in (1,2)):
                raise ValueError("Unrecognized lat/lon dimensionality")
            # Only inspect coordinate metadata and *two scalar time values*;
            # never read the full time vector or a temperature grid.
            start = float(time.isel(time=0).values)
            end = float(time.isel(time=-1).values)
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                raise ValueError("Invalid time-coordinate endpoints")
            time_units = str(time.attrs.get("units", ""))
            if "since" not in time_units:
                raise ValueError("CF time units missing 'since' anchor")
            return {
                "status": "opened_and_schema_checked_metadata_only",
                "store": store,
                "variable": "tas",
                "tas_units": units,
                "time_calendar": calendar,
                "time_units": time_units,
                "first_time_cf_value": start,
                "last_time_cf_value": end,
                "n_time_steps": int(time.sizes["time"]),
                "tas_dims": list(tas.dims),
                "tas_shape": list(tas.shape),
                "lat_shape": list(lat.shape),
                "lon_shape": list(lon.shape),
                "lat_dims": list(lat.dims),
                "lon_dims": list(lon.dims),
                "dtype": str(tas.dtype),
                "climate_fields_loaded": False,
                "notes": "Scalar time endpoints in native numeric CF units only; actual calendar dates not yet decoded",
            }
        finally:
            ds.close()
    except Exception as exc:
        # Do not manufacture an accessible store or silently replace it.
        return {"status": "unverified_store_or_schema",
                "store": store, "failure_type": type(exc).__name__,
                "error": str(exc)[:600]}


def audit(manifest: dict, max_models: int = 2) -> dict:
    if (manifest.get("status") != "catalog_metadata_only_no_climate_fields_or_transfer_skill"
            or manifest.get("variables") != ["tas"] or manifest.get("table_id") != "day"):
        raise ValueError("Require verified daily tas metadata manifest")
    candidates = manifest.get("selected_candidates", [])
    if not 1 <= max_models <= 3 or len(candidates) < max_models:
        raise ValueError("Require 1–3 bounded candidate source models")
    results = []
    for model in candidates[:max_models]:
        trio = {key: inspect_store(model[key]["store"])
                for key in ("historical","ssp245","ssp585")}
        eligible = all(item["status"] == "opened_and_schema_checked_metadata_only"
                       for item in trio.values())
        if eligible:
            units = {item["tas_units"] for item in trio.values()}
            calendars = {item["time_calendar"] for item in trio.values()}
            shapes = {tuple(item["tas_shape"][1:]) for item in trio.values()}
            eligible = len(units) == len(calendars) == len(shapes) == 1
        results.append({
            "source_id":model["source_id"],
            "member_id":model["member_id"],
            "grid_label":model["grid_label"],
            "matched_trio_accessible_and_compatible":bool(eligible),
            "stores":trio,
        })
    return {
        "status":"actual_public_zarr_metadata_probe_no_forecast_or_future_skill",
        "candidate_catalog_sha256":manifest["catalog_sha256"],
        "examined_model_count":len(results),
        "complete_trios":sum(x["matched_trio_accessible_and_compatible"] for x in results),
        "models":results,
        "remaining_gates":[
            "Decode exact actual calendar-based date ranges; verify historical and future SSP period bounds",
            "Verify data slice and missing values with tiny explicitly bounded spatial/temporal reads",
            "Select consistent model-native physical grid and daily-mean +1 day target definition",
            "Prespecify training/validation split and freeze loss selection without SSP leakage",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--max-models",type=int,default=2)
    args = parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    manifest=json.loads(Path(args.manifest).read_text())
    report=audit(manifest,args.max_models)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":report["status"],"examined_model_count":report["examined_model_count"],
        "complete_trios":report["complete_trios"],
        "models":[{"name":x["source_id"],
                   "complete":x["matched_trio_accessible_and_compatible"],
                   "statuses":{k:v["status"] for k,v in x["stores"].items()}}
                  for x in report["models"]]},indent=2))


if __name__=="__main__":
    main()
