"""Tiny, bounded AWI-CM-1-1-MR native-grid daily TAS regional-pair audit.

This is NOT a training dataset or future model evaluation. Read no more than
two daily regional grids at two predeclared dates per scenario. Preserve native
CMIP6 CF calendar and explicitly test one-day lead across each source/target.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

LATS = (25.0, 50.0)
LONS_EAST = (235.0, 260.0)  # 125W–100W, no longitudinal interpolation
SCHEDULE = {
    "historical": ((2010, 1, 15), (2010, 7, 15)),
    "ssp245": ((2030, 1, 15), (2050, 7, 15)),
    "ssp585": ((2030, 1, 15), (2050, 7, 15)),
}


def contiguous_span(values, low: float, high: float):
    import numpy as np
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError("Regional read requires finite one-dimensional native coordinate")
    ids = np.flatnonzero((arr >= low) & (arr <= high))
    if len(ids) < 4 or not np.all(np.diff(ids) == 1):
        raise ValueError("Regional coordinates require >=4 consecutive grid cells")
    return slice(int(ids[0]), int(ids[-1]) + 1)


def exact_daily_pair_indices(numeric_times, date, units, calendar):
    import cftime
    import numpy as np
    times = np.asarray(numeric_times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("Invalid native time coordinates")
    dt = cftime.datetime(*date, 12, 0, 0, calendar=calendar)
    target = float(cftime.date2num(dt, units=units, calendar=calendar))
    index = int(np.searchsorted(times, target))
    if index + 1 >= len(times) or abs(float(times[index]) - target) > 1e-5:
        raise ValueError("Predeclared source date not an exact native daily coordinate")
    first, second = cftime.num2date([times[index], times[index + 1]],
                                   units=units,calendar=calendar,
                                   only_use_cftime_datetimes=True)
    delta = second - first
    if delta.days != 1 or delta.seconds != 0:
        raise ValueError("Source–target time lead not exactly one native day")
    return index, index + 1, str(first), str(second)


def audit_store(store: str, experiment: str) -> dict:
    if experiment not in SCHEDULE or not store.startswith("gs://cmip6/"):
        raise ValueError("Unaudited CMIP6 experiment or store")
    import fsspec
    import numpy as np
    import xarray as xr
    mapper = fsspec.get_mapper(store, anon=True, token="anon")
    ds = xr.open_zarr(mapper, consolidated=True, decode_times=False,
                      chunks=None, mask_and_scale=True)
    try:
        tas=ds["tas"]
        if tas.attrs.get("units")!="K" or tas.ndim!=3 or set(tas.dims)!={"time","lat","lon"}:
            raise ValueError("Require native daily Kelvin TAS on simple regular lat/lon grid")
        lat=np.asarray(ds["lat"].values,dtype=float)
        lon=np.asarray(ds["lon"].values,dtype=float) % 360
        lat_slice=contiguous_span(lat,*LATS)
        lon_slice=contiguous_span(lon,*LONS_EAST)
        time=ds["time"]
        calendar=str(time.attrs.get("calendar",time.encoding.get("calendar","standard")))
        units=str(time.attrs.get("units",""))
        if "since" not in units: raise ValueError("No CF time units")
        # Reading one 1D numeric coordinate of up to ~60k elements, NOT any global field.
        times=np.asarray(time.values,dtype=float)
        reports=[]
        for dt in SCHEDULE[experiment]:
            first,second,source_date,target_date=exact_daily_pair_indices(times,dt,units,calendar)
            source=np.asarray(tas.isel(time=first,lat=lat_slice,lon=lon_slice).values,dtype=float)
            target=np.asarray(tas.isel(time=second,lat=lat_slice,lon=lon_slice).values,dtype=float)
            if source.shape!=target.shape or source.ndim!=2:
                raise ValueError("Source/target region shape mismatch")
            if source.size>1600 or min(source.shape)<4:
                raise ValueError("Regional pilot is larger than 1600 cells or too small")
            if not np.isfinite(source).all() or not np.isfinite(target).all():
                raise ValueError("Tiny regional pilot has missing temperature cells")
            if min(float(source.min()),float(target.min()))<150 or max(float(source.max()),float(target.max()))>350:
                raise ValueError("Implausible Kelvin in pilot regional grids")
            weights=np.cos(np.deg2rad(lat[lat_slice]))[:,None]
            rmse=float(np.sqrt(np.sum(weights*(target-source)**2)/(weights.sum()*target.shape[1])))
            reports.append({
                "source_date":source_date,"target_date":target_date,
                "n_cells":int(source.size),"latitude_count":int(source.shape[0]),
                "longitude_count":int(source.shape[1]),
                "source_min_K":float(source.min()),"source_max_K":float(source.max()),
                "target_min_K":float(target.min()),"target_max_K":float(target.max()),
                "persistence_rmse_K_single_pair_descriptive_only":rmse,
                "sampled_cells_finite":True,
            })
        return {"status":"real_native_cmip6_1day_regional_pairs_very_small_pilot",
                "store":store,"calendar":calendar,"time_units":units,
                "latitude_range_native":[float(lat[lat_slice][0]),float(lat[lat_slice][-1])],
                "longitude_range_native":[float(lon[lon_slice][0]),float(lon[lon_slice][-1])],
                "regional_grid_shape":[lat_slice.stop-lat_slice.start,lon_slice.stop-lon_slice.start],
                "pairs":reports,"raw_temperature_grids_archived":False}
    finally:
        ds.close()


def run(manifest: dict)->dict:
    if manifest.get("status")!="catalog_metadata_only_no_climate_fields_or_transfer_skill":
        raise ValueError("Missing catalog audit")
    awi=[m for m in manifest["selected_candidates"]
         if m["source_id"]=="AWI-CM-1-1-MR" and m["member_id"]=="r1i1p1f1" and m["grid_label"]=="gn"]
    if len(awi)!=1:
        raise ValueError("Exactly one verified AWI-CM-1-1-MR model-member-grid expected")
    m=awi[0]
    results={exp:audit_store(m[exp]["store"],exp) for exp in SCHEDULE}
    shapes={tuple(r["regional_grid_shape"]) for r in results.values()}
    calendars={r["calendar"] for r in results.values()}
    if len(shapes)!=1 or len(calendars)!=1:
        raise ValueError("Daily scenario grids/calendars not compatible")
    return {"status":"small_real_awi_daily_pair_preflight_not_training_or_future_skill",
            "model":"AWI-CM-1-1-MR","member":"r1i1p1f1","grid":"gn",
            "source_catalog_sha256":manifest["catalog_sha256"],
            "target_definition":"tas_daily_mean_at_t_to_t_plus_one_native_day",
            "spatial_region":"25–50N 125–100W native grid no interpolation",
            "scenarios":results,
            "limitations":["Only two sparse pilot forecast pairs per scenario",
                           "Persistence RMSE is a pipeline sanity check, not scientific SSP forecast skill",
                           "No historical fitted model, CMIP6 forecast training, or global warming attribution",
                           "No complete spatial/time missingness audit or multi-model validation"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    report=run(json.loads(Path(args.manifest).read_text()))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"model":report["model"],
        "scenarios":{k:{"shape":v["regional_grid_shape"],
                       "pairs":len(v["pairs"]),
                       "date_leads":[(p["source_date"],p["target_date"]) for p in v["pairs"]]}
                       for k,v in report["scenarios"].items()}},indent=2))


if __name__=="__main__":
    main()
