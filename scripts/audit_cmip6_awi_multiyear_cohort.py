"""Bounded multi-year, native-calendar AWI daily tas paired data cohort audit.

This v4.4 audit collects *only summary statistics* for a fixed historical/SSP
cohort. It does not train model weights or select a loss using SSP targets.
The sample is sparse and must never be described as continuous daily coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from audit_cmip6_awi_daily_pairs import contiguous_span, exact_daily_pair_indices

TRAIN_YEARS = (1980, 1990, 2000, 2004)
VALID_YEARS = (2005, 2010, 2014)
SSP_YEARS = (2030, 2050, 2080)
MONTH_DAYS = (8, 20)
TRAIN = "historical"
SCENARIOS = ("ssp245", "ssp585")
SOURCE_ID = "AWI-CM-1-1-MR"
MEMBER = "r1i1p1f1"
GRID = "gn"


def expected_schedule():
    samples = {}
    for experiment, folds in (
        ("historical", (("train", TRAIN_YEARS), ("validation", VALID_YEARS))),
        ("ssp245", (("future_test", SSP_YEARS),)),
        ("ssp585", (("future_test", SSP_YEARS),)),
    ):
        dates = []
        for fold, years in folds:
            for year in years:
                for month in range(1, 13):
                    for day in MONTH_DAYS:
                        dates.append((fold, year, month, day))
        samples[experiment] = dates
    return samples


def validate_schedule(plan):
    if set(plan) != {"historical", "ssp245", "ssp585"}:
        raise ValueError("All three source experiments required")
    train = {y for fold, y, _, _ in plan["historical"] if fold == "train"}
    valid = {y for fold, y, _, _ in plan["historical"] if fold == "validation"}
    if train != set(TRAIN_YEARS) or valid != set(VALID_YEARS):
        raise ValueError("Unexpected historical temporal folds")
    if max(train) >= min(valid):
        raise ValueError("Historical training/validation overlap")
    for scenario in SCENARIOS:
        if any(fold != "future_test" or yr not in SSP_YEARS for fold, yr, _, _ in plan[scenario]):
            raise ValueError("SSP sample leaked into model-fitting split")
    if sum(len(v) for v in plan.values()) > 350:
        raise ValueError("Bounded pilot exceeds 350 daily source-target pairs")
    for exp, rows in plan.items():
        if len(rows) != len(set(rows)):
            raise ValueError("Duplicate dates in schedule")


def verify_member(manifest):
    if manifest.get("status") != "catalog_metadata_only_no_climate_fields_or_transfer_skill":
        raise ValueError("Require audited model catalog")
    candidates = [m for m in manifest["selected_candidates"]
                  if (m["source_id"], m["member_id"], m["grid_label"])
                  == (SOURCE_ID, MEMBER, GRID)]
    if len(candidates) != 1:
        raise ValueError("Require exactly one audited AWI ensemble match")
    m = candidates[0]
    for exp in ("historical", *SCENARIOS):
        if not m[exp]["store"].startswith("gs://cmip6/"):
            raise ValueError("Unexpected source store")
    return m


def audit_one_store(store, scheduled):
    import fsspec
    import numpy as np
    import xarray as xr
    mapper = fsspec.get_mapper(store, anon=True, token="anon")
    ds = xr.open_zarr(mapper, consolidated=True, decode_times=False,
                      chunks=None, mask_and_scale=True)
    try:
        tas = ds["tas"]
        if tas.ndim != 3 or set(tas.dims) != {"time", "lat", "lon"} or tas.attrs.get("units") != "K":
            raise ValueError("Expected 3D native daily mean Kelvin tas")
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float) % 360
        lat_sel = contiguous_span(lat, 25, 50)
        lon_sel = contiguous_span(lon, 235, 260)
        grid_shape = (lat_sel.stop-lat_sel.start, lon_sel.stop-lon_sel.start)
        if not (16 <= grid_shape[0]*grid_shape[1] <= 1600):
            raise ValueError("Unexpected regional grid shape")
        month_counts = Counter()
        fold_counts = Counter()
        time = ds["time"]
        units = time.attrs.get("units", "")
        calendar = time.attrs.get("calendar", time.encoding.get("calendar", "standard"))
        if "since" not in units:
            raise ValueError("Invalid CF time units")
        numeric_time = np.asarray(time.values, dtype=float)
        if np.any(np.diff(numeric_time) <= 0):
            raise ValueError("Nonascending native calendar")
        weights = np.cos(np.deg2rad(lat[lat_sel]))[:,None]
        weighted_sum_sq = 0.0
        weighted_sum = 0.0
        x_mean_sum = 0.0
        y_mean_sum = 0.0
        seen = []
        for fold, year, month, day in scheduled:
            src_ix, target_ix, src_date, dst_date = exact_daily_pair_indices(
                numeric_time,(year,month,day),units,calendar)
            source = np.asarray(tas.isel(time=src_ix,lat=lat_sel,lon=lon_sel).values,dtype=float)
            target = np.asarray(tas.isel(time=target_ix,lat=lat_sel,lon=lon_sel).values,dtype=float)
            if source.shape != grid_shape or target.shape != grid_shape:
                raise ValueError("Unexpected source/target spatial shape")
            if not np.isfinite(source).all() or not np.isfinite(target).all():
                raise ValueError(f"Missing Kelvin value at {src_date}")
            if not (150 <= min(source.min(),target.min())
                    and max(source.max(),target.max()) <= 350):
                raise ValueError("Physically implausible sampled Kelvin")
            if int(dst_date[:4]) != year or int(src_date[:4]) != year:
                raise ValueError("Cross-year target leakage or missing day")
            d = target-source
            weighted_sum_sq += float(np.sum(weights * d**2))
            weighted_sum += float(weights.sum() * source.shape[1])
            x_mean_sum += float(source.mean())
            y_mean_sum += float(target.mean())
            month_counts[month] += 1
            fold_counts[fold] += 1
            seen.append((src_date, dst_date))
        if len(set(seen)) != len(seen):
            raise ValueError("Duplicate forecast pair")
        if len(month_counts) != 12 or set(month_counts.values()) != {
                len(scheduled)//12}:
            raise ValueError("Incomplete or unequal month coverage")
        return {
            "status": "real_multiple_year_sparse_native_daily_pairs_checked",
            "store":store,"calendar":str(calendar),"time_units":str(units),
            "regional_grid_shape":list(grid_shape),
            "latitude_degree_samples":[float(lat[lat_sel][0]),float(lat[lat_sel][-1])],
            "longitude_east_degree_samples":[float(lon[lon_sel][0]),float(lon[lon_sel][-1])],
            "n_pairs":len(seen),"per_month_counts":{str(m):month_counts[m] for m in range(1,13)},
            "per_fold_counts":dict(fold_counts),
            "first_pair":list(seen[0]),"last_pair":list(seen[-1]),
            "overall_persistence_weighted_rmse_K_descriptive_pilot":(weighted_sum_sq/weighted_sum)**0.5,
            "mean_source_temperature_K":x_mean_sum/len(seen),
            "mean_target_temperature_K":y_mean_sum/len(seen),
            "finite_sampled_grids":True,
            "raw_data_archived":False,
        }
    finally:
        ds.close()


def run(manifest):
    plan=expected_schedule()
    validate_schedule(plan)
    m=verify_member(manifest)
    result={}
    for exp, scheduled in plan.items():
        result[exp]=audit_one_store(m[exp]["store"],scheduled)
    shapes={tuple(v["regional_grid_shape"]) for v in result.values()}
    calendars={v["calendar"] for v in result.values()}
    if len(shapes)!=1 or len(calendars)!=1:
        raise ValueError("Scenarios disagree on grid or native calendar")
    assert result["historical"]["per_fold_counts"] == {"train":96, "validation":72}
    for scenario in SCENARIOS:
        assert result[scenario]["per_fold_counts"] == {"future_test":72}
    schedule_repr=json.dumps(plan,sort_keys=True,separators=(",",":"))
    return {
        "status":"v44_sparse_multiyear_data_audit_no_neural_forecasting",
        "source_id":SOURCE_ID,"member_id":MEMBER,"grid_label":GRID,
        "catalog_sha256":manifest["catalog_sha256"],
        "schedule_sha256":hashlib.sha256(schedule_repr.encode()).hexdigest(),
        "schedule":{"train_years":list(TRAIN_YEARS),"validation_years":list(VALID_YEARS),
                    "ssp_test_years":list(SSP_YEARS),"month_days":list(MONTH_DAYS)},
        "n_pairs_total":sum(x["n_pairs"] for x in result.values()),
        "experiments":result,
        "limitations":[
            "Only two fixed dates per month per selected year; not continuous daily coverage",
            "No neural training, model validation, tail-loss comparison, or future forecast skill",
            "No future SSP measurements outside this sparse sample",
            "No source climatology/p95 learned from future data",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    output=Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    result=run(json.loads(Path(a.manifest).read_text()))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":result["status"],"n_pairs_total":result["n_pairs_total"],
        "models":{k:{"n_pairs":v["n_pairs"],"shape":v["regional_grid_shape"],
                     "month_counts":v["per_month_counts"],
                     "persistence_rmse_K":v["overall_persistence_weighted_rmse_K_descriptive_pilot"]}
                  for k,v in result["experiments"].items()}},indent=2))


if __name__=="__main__":
    main()
