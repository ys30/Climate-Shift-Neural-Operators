"""Build v4.5 fixed historical AWI-CMIP6 native daily tas pairs for later model training.

Historical only: no SSP test targets ever enter normalization, threshold fitting
or checkpoint choice. This is a 3-days/month 1980–2004 train and 2-days/month
2005–2014 validation *sparse daily* benchmark, not continuous heatwave data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from audit_cmip6_awi_daily_pairs import contiguous_span, exact_daily_pair_indices

TRAIN_YEARS = tuple(range(1980, 2005))
VALID_YEARS = tuple(range(2005, 2015))
TRAIN_DAYS = (6, 16, 26)
VALID_DAYS = (8, 20)
MODEL = ("AWI-CM-1-1-MR","r1i1p1f1","gn")

def schedule():
    return [(fold,year,month,day)
            for fold,years,days in (("train",TRAIN_YEARS,TRAIN_DAYS),
                                    ("validation",VALID_YEARS,VALID_DAYS))
            for year in years for month in range(1,13) for day in days]

def validate_plan(plan):
    if len(plan)!=1140 or len(set(plan))!=1140:
        raise ValueError("Frozen 1140 samples with unique dates required")
    counts = Counter((fold,month) for fold,year,month,day in plan)
    if any(counts["train",m]!=75 or counts["validation",m]!=20 for m in range(1,13)):
        raise ValueError("Train month needs 75 and validation 20 exact dates")
    for fold,year,month,day in plan:
        if not (1<=month<=12 and 1<=day<=26):
            raise ValueError("Invalid calendar date")
        if fold=="train" and (year not in TRAIN_YEARS or day not in TRAIN_DAYS):
            raise ValueError("Train uses only predeclared historical years/days")
        if fold=="validation" and (year not in VALID_YEARS or day not in VALID_DAYS):
            raise ValueError("Validation uses only predeclared historical years/days")
        if fold not in ("train","validation"):
            raise ValueError("SSP future years must not appear in historical fit cohort")
    if max(TRAIN_YEARS)>=min(VALID_YEARS):
        raise ValueError("Training overlaps validation")

def validate_manifest(manifest):
    if manifest.get("status")!="catalog_metadata_only_no_climate_fields_or_transfer_skill":
        raise ValueError("Unverified catalog")
    found=[m for m in manifest["selected_candidates"]
           if (m["source_id"],m["member_id"],m["grid_label"])==MODEL]
    if len(found)!=1:
        raise ValueError("Missing exactly-one verified AWI Historical source")
    url=found[0]["historical"]["store"]
    if not url.startswith("gs://cmip6/") or "/historical/" not in url or "/day/tas/" not in url:
        raise ValueError("Training must use Historical daily tas only")
    return url

def extract(manifest, outfile):
    import fsspec
    import numpy as np
    import xarray as xr
    from climate_shift.data import load_npz_pairs

    plan=schedule()
    validate_plan(plan)
    store=validate_manifest(manifest)
    mapper=fsspec.get_mapper(store,anon=True,token="anon")
    ds=xr.open_zarr(mapper,consolidated=True,decode_times=False,
                    chunks=None,mask_and_scale=True)
    try:
        tas=ds["tas"]
        if tas.attrs.get("units")!="K" or set(tas.dims)!={"time","lat","lon"}:
            raise ValueError("Native daily Kelvin tas(time,lat,lon) required")
        latitude=np.asarray(ds["lat"].values,dtype=float)
        longitude=np.asarray(ds["lon"].values,dtype=float)%360
        lat_sel=contiguous_span(latitude,25.,50.)
        lon_sel=contiguous_span(longitude,235.,260.)
        regional_lats=latitude[lat_sel].astype(np.float64)
        regional_lons=longitude[lon_sel].astype(np.float64)
        h,w=len(regional_lats),len(regional_lons)
        if (h,w)!=(26,27):
            raise ValueError(f"Source model native grid changed: {(h,w)}")
        time=ds["time"]
        calendar=str(time.attrs.get("calendar",time.encoding.get("calendar","standard")))
        units=str(time.attrs.get("units",""))
        if calendar not in ("proleptic_gregorian","gregorian","standard"):
            raise ValueError("This datetime64-compatible exporter excludes non-Gregorian calendars")
        if "since" not in units:
            raise ValueError("Unrecognized native time units")
        times=np.asarray(time.values,dtype=float)
        if not np.all(np.diff(times)>0):
            raise ValueError("Native time coordinate not monotone")
        xs,ys,src_dates,dst_dates,yrs=[],[],[],[],[]
        fold_counts=Counter()
        for fold,year,month,day in plan:
            i,j,source_date,target_date=exact_daily_pair_indices(times,(year,month,day),units,calendar)
            if source_date[:10]!=f"{year:04d}-{month:02d}-{day:02d}":
                raise ValueError("Native date mismatch")
            if target_date[:4]!=str(year):
                raise ValueError("Source-target crossed year boundary")
            source=tas.isel(time=i,lat=lat_sel,lon=lon_sel).values
            target=tas.isel(time=j,lat=lat_sel,lon=lon_sel).values
            source=np.asarray(source,dtype=np.float32)
            target=np.asarray(target,dtype=np.float32)
            if source.shape!=(h,w) or target.shape!=(h,w):
                raise ValueError("Source/target regional shape mismatch")
            if not np.isfinite(source).all() or not np.isfinite(target).all():
                raise ValueError(f"Non-finite value in {source_date}")
            if min(source.min(),target.min())<150 or max(source.max(),target.max())>350:
                raise ValueError("Invalid Kelvin in sampled source or target")
            xs.append(source[None]);ys.append(target[None])
            src_dates.append(np.datetime64(source_date[:10]+"T12:00:00"))
            dst_dates.append(np.datetime64(target_date[:10]+"T12:00:00"))
            yrs.append(year)
            fold_counts[fold]+=1
        x=np.stack(xs).astype(np.float32)
        y=np.stack(ys).astype(np.float32)
        source_times=np.asarray(src_dates,dtype="datetime64[ns]")
        target_times=np.asarray(dst_dates,dtype="datetime64[ns]")
        years=np.asarray(yrs,dtype=np.int32)
        if not np.all(target_times-source_times==np.timedelta64(1,"D")):
            raise ValueError("Daily target is not exact +1 day")
        train=years<=2004;valid=(years>=2005)&(years<=2014)
        if (train.sum(),valid.sum())!=(900,240) or not target_times[train].max()<source_times[valid].min():
            raise ValueError("Train/validation source–target boundary contamination")
        if any(np.sum((years[train]>=1980) & (target_times[train].astype("datetime64[M]").astype(int)%12==m))!=75 for m in range(12)):
            raise ValueError("Insufficient training calendar-month P95 sample size")
        outfile.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(outfile,X=x,y=y,years=years,
                            latitude=regional_lats,source_times=source_times,target_times=target_times)
        read_x,read_y,read_years=load_npz_pairs(outfile)
        if read_x.shape!=x.shape or read_y.shape!=y.shape or not np.array_equal(read_years,years):
            raise ValueError("Saved dataset failed round-trip schema check")
        # The existing chronological_masks requires a third fold; for an honest
        # historical-only train/validation cohort, isolate the two folds directly.
        from climate_shift.seasonal_extremes import fit_local_monthly_p95
        thresholds,reference=fit_local_monthly_p95(y[train],target_times[train],min_samples_per_month=20)
        if thresholds.shape!=(12,h,w):
            raise ValueError("Historical-only monthly P95 shape incorrect")
        weights=np.cos(np.deg2rad(regional_lats))[:,None]
        val_error=(x[valid]-y[valid])[:,0]
        persistence_rmse=float(np.sqrt(np.sum(weights[None]*val_error**2)
                                       /(weights.sum()*w*valid.sum())))
        return {
            "status":"v45_historical_dense_sparsely_sampled_training_ready_no_model_training",
            "source":store,"model":MODEL[0],"member":MODEL[1],"grid":MODEL[2],
            "source_catalog_sha256":manifest["catalog_sha256"],
            "train_years":[1980,2004],"validation_years":[2005,2014],
            "sample_days_per_month":{"train":list(TRAIN_DAYS),"validation":list(VALID_DAYS)},
            "samples_per_calendar_month":{"train":75,"validation":20},
            "sample_count":{"train":int(train.sum()),"validation":int(valid.sum()),"total":len(years)},
            "source_variable":"daily_mean_tas_K","native_calendar":calendar,
            "region_grid_shape":[h,w],"latitude_degrees":regional_lats.tolist(),
            "longitude_east_bounds":[float(regional_lons[0]),float(regional_lons[-1])],
            "first_source_time":str(source_times[0]),"last_target_time":str(target_times[-1]),
            "forecast_horizon_hours":24,
            "train_only_local_monthly_p95":"fit_passed_75_targets_per_month",
            "train_only_p95_reference":reference,
            "validation_persistence_weighted_rmse_K_descriptive":persistence_rmse,
            "dataset_sha256":hashlib.sha256(outfile.read_bytes()).hexdigest(),
            "schedule_sha256":hashlib.sha256(json.dumps(plan).encode()).hexdigest(),
            "NPZ_contains_only_historical":True,
            "not_continuous_daily":True,
            "ssp_test_data_read":False,
            "forecast_model_trained":False,
            "caveats":[
                "Three dates per month per training year, not complete daily sequences",
                "P95 per month based on 75 targets from 25 simulated years, not observed future climate",
                "No neural forecast skill or climate-change signal fidelity yet",
                "CMIP6 daily-mean task differs from ERA5 instantaneous +24h task",
            ],
        }
    finally:
        ds.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    path=Path(args.output)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("Refuse to overwrite existing experiment data")
    report=extract(json.loads(Path(args.manifest).read_text()),path)
    path.with_suffix(".json").write_text(json.dumps(report,indent=2,default=str)+"\n")
    print(json.dumps({k:report[k] for k in ("status","sample_count",
        "samples_per_calendar_month","region_grid_shape",
        "train_only_local_monthly_p95",
        "validation_persistence_weighted_rmse_K_descriptive",
        "dataset_sha256")},indent=2))

if __name__=="__main__":
    main()
