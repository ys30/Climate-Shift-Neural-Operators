"""Read-only CF-calendar and tiny-slice audit for verified public CMIP6 daily tas.

This audit deliberately does not train a model, fetch entire climate grids,
regrid or interpret CMIP projections as observed future temperatures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe_cmip6_zarr import inspect_store


def audit_store(store: str) -> dict:
    base = inspect_store(store)
    if base["status"] != "opened_and_schema_checked_metadata_only":
        return {"status": "failed_metadata_gate", "metadata": base}

    try:
        import cftime
        import fsspec
        import numpy as np
        import xarray as xr

        mapper = fsspec.get_mapper(store, anon=True, token="anon")
        ds = xr.open_zarr(mapper, consolidated=True, decode_times=False,
                          chunks=None, mask_and_scale=True)
        try:
            time = ds["time"]
            units = base["time_units"]
            calendar = base["time_calendar"]
            raw_first = time.isel(time=0).values.item()
            raw_last = time.isel(time=-1).values.item()
            decoded = cftime.num2date(
                [raw_first, raw_last], units=units, calendar=calendar,
                only_use_cftime_datetimes=True)
            dates = [str(value) for value in decoded]
            start_year, end_year = int(decoded[0].year), int(decoded[1].year)
            if start_year >= end_year:
                raise ValueError("Calendar decode did not produce increasing years")

            latitude = ds["lat"]
            longitude = ds["lon"]
            # Bound read to four scalar coordinates only, not spatial arrays.
            def sample_coordinate(coord):
                index_first = {dim:0 for dim in coord.dims}
                index_last = {dim:coord.sizes[dim]-1 for dim in coord.dims}
                return [float(coord.isel(index_first).values),
                        float(coord.isel(index_last).values)]
            lat_samples = sample_coordinate(latitude)
            lon_samples = sample_coordinate(longitude)
            if not all(np.isfinite(x) for x in lat_samples+lon_samples):
                raise ValueError("Nonfinite coordinate edge samples")
            if any(abs(x)>90.0001 for x in lat_samples):
                raise ValueError("Latitude samples outside physical range")

            tas = ds["tas"]
            # Only one tiny temperature sample: first / midpoint / last time,
            # each at up to 2×2 grid cells. No large whole-field download.
            spatial_dims = [dim for dim in tas.dims if dim != "time"]
            if len(spatial_dims)!=2:
                raise ValueError("Expected two spatial dims in daily tas")
            h,w=(int(tas.sizes[dim]) for dim in spatial_dims)
            sample_indices = sorted(set([0, len(time)//2, len(time)-1]))
            tiny_samples = []
            for t in sample_indices:
                sl=tas.isel(time=t, **{spatial_dims[0]:slice(0,min(2,h)),
                                      spatial_dims[1]:slice(0,min(2,w))})
                values=np.asarray(sl.values, dtype=float)
                finite=values[np.isfinite(values)]
                tiny_samples.append({
                    "time_index":t, "n_finite":int(len(finite)),
                    "n_cells":int(values.size),
                    "min_K":float(finite.min()) if len(finite) else None,
                    "max_K":float(finite.max()) if len(finite) else None,
                })
            if not any(x["n_finite"] for x in tiny_samples):
                raise ValueError("All bounded sampled tas grids missing")
            if any(x["min_K"] is not None and (x["min_K"]<150 or x["max_K"]>350)
                   for x in tiny_samples):
                raise ValueError("Implausible native CMIP6 Kelvin sample")

            return {
                "status":"small_slice_cf_dates_and_kelvin_samples_verified",
                "store":store,
                "n_time_steps":int(time.sizes["time"]),
                "start_date_native_calendar":dates[0],
                "end_date_native_calendar":dates[1],
                "start_year":start_year,"end_year":end_year,
                "calendar":calendar,"time_units":units,
                "tas_units":base["tas_units"],
                "spatial_dims":spatial_dims,
                "spatial_grid_shape":[h,w],
                "lat_first_last_samples":lat_samples,
                "lon_first_last_samples":lon_samples,
                "tiny_temperatures":tiny_samples,
                "full_fields_loaded":False,
                "n_requested_temperature_gridcells_max":3*2*2,
            }
        finally:
            ds.close()
    except Exception as exc:
        return {"status":"failed_small_slice_or_cf_audit","store":store,
                "exception_type":type(exc).__name__,"error":str(exc)[:700]}


def audit_manifest(manifest:dict, count:int=2)->dict:
    if manifest.get("status")!="catalog_metadata_only_no_climate_fields_or_transfer_skill":
        raise ValueError("Require successful strict catalog metadata input")
    if not 1<=count<=2 or len(manifest.get("selected_candidates",[]))<count:
        raise ValueError("Require one or two catalog-matched model trios")
    models=[]
    for candidate in manifest["selected_candidates"][:count]:
        scenarios={exp:audit_store(candidate[exp]["store"])
                   for exp in ("historical","ssp245","ssp585")}
        ok=all(s["status"]=="small_slice_cf_dates_and_kelvin_samples_verified"
               for s in scenarios.values())
        if ok:
            hist=scenarios["historical"]
            s245=scenarios["ssp245"]
            s585=scenarios["ssp585"]
            ok=(hist["end_year"]<=2015
                and s245["start_year"]>=2014
                and s585["start_year"]>=2014
                and hist["calendar"]==s245["calendar"]==s585["calendar"]
                and hist["spatial_grid_shape"]==s245["spatial_grid_shape"]==s585["spatial_grid_shape"]
                and all(s["tas_units"] in ("K","Kelvin","kelvin")
                        for s in scenarios.values()))
        models.append({
            "source_id":candidate["source_id"],
            "member_id":candidate["member_id"],
            "grid_label":candidate["grid_label"],
            "historical_ssp_daily_schema_small_slice_compatible":bool(ok),
            "experiments":scenarios,
        })
    return {"status":"bounded_actual_cmip6_small_slice_audit_no_model_training",
            "catalog_sha256":manifest["catalog_sha256"],
            "model_count":len(models),
            "compatible_trios":sum(m["historical_ssp_daily_schema_small_slice_compatible"] for m in models),
            "models":models,
            "limitations":[
                "Only 12 temperature gridcells per store maximum: not a whole-grid missingness audit",
                "No guarantee continuity, spatial geometry, grid-cell validity or archive completeness",
                "No precipitation, winds, tasmax or instantaneous ERA5 equivalence checked",
                "Scenario outputs are projections, not independently observed future truth",
                "No future climate neural operator skill has been evaluated"]}


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--max-models",type=int,default=2)
    args=p.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    data=json.loads(Path(args.manifest).read_text())
    result=audit_manifest(data,args.max_models)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"compatible_trios":result["compatible_trios"],
        "models":[{"name":x["source_id"],"compatible":x["historical_ssp_daily_schema_small_slice_compatible"],
                   "dates":{k:(v.get("start_date_native_calendar"),
                               v.get("end_date_native_calendar"),
                               v.get("status")) for k,v in x["experiments"].items()}}
                  for x in result["models"]]},indent=2))


if __name__=="__main__":
    main()
