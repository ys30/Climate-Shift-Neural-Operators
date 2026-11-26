"""Export BCC native noleap Historical-only daily +1day forecast pairs.

Do NOT serialize noleap times as Gregorian datetime64 or reuse AWI normalizers.
Frozen train: 1980-2004 days 6,16,26 each month; val: 2005-2014 days
8,20 each month. BCC source grid 23x23 and independently fitted local P95.
No future SSP data is touched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .audit_cmip6_awi_daily_pairs import contiguous_span, exact_daily_pair_indices
    from .audit_cmip6_bcc_regional_pairs import get_model
except ImportError:
    from audit_cmip6_awi_daily_pairs import contiguous_span, exact_daily_pair_indices
    from audit_cmip6_bcc_regional_pairs import get_model

TRAIN_YEARS = tuple(range(1980, 2005))
VALID_YEARS = tuple(range(2005, 2015))
TRAIN_DAYS = (6, 16, 26)
VALID_DAYS = (8, 20)


def schedule():
    rows=[]
    for fold, years, days in (
        ("train", TRAIN_YEARS, TRAIN_DAYS),
        ("validation", VALID_YEARS, VALID_DAYS),
    ):
        rows.extend((fold, year, month, day)
                    for year in years for month in range(1, 13) for day in days)
    return rows


def validate_schedule(rows):
    if len(rows)!=1140 or len(set(rows))!=1140:
        raise ValueError("Expected unique 1140 BCC Historical-only dates")
    from collections import Counter
    counts=Counter((fold,month) for fold,year,month,day in rows)
    if any(counts["train",m]!=75 or counts["validation",m]!=20 for m in range(1,13)):
        raise ValueError("Unfrozen monthly sample coverage")
    for fold,year,month,day in rows:
        if fold=="train":
            if year not in TRAIN_YEARS or day not in TRAIN_DAYS:
                raise ValueError("Invalid train year or day")
        elif fold=="validation":
            if year not in VALID_YEARS or day not in VALID_DAYS:
                raise ValueError("Invalid validation year or day")
        else:
            raise ValueError("Do not incorporate SSP training/test years")
    if max(TRAIN_YEARS)>=min(VALID_YEARS):
        raise ValueError("Boundary contamination")


def native_noleap_pair(times, year, month, day, units, calendar):
    i,j,src,target=exact_daily_pair_indices(times,(year,month,day),units,calendar)
    if (src[:10]!=f"{year:04d}-{month:02d}-{day:02d}"
            or target[:4]!=str(year)):
        raise ValueError("Source/target native-calendar misalignment")
    return i,j,src,target


def monthly_training_reference(y, years, months, train):
    if y.ndim!=4 or y.shape[1]!=1 or not np.isfinite(y).all():
        raise ValueError("Expected finite 4-D Kelvin temperature targets")
    if years.shape!=(len(y),) or months.shape!=(len(y),) or train.shape!=(len(y),):
        raise ValueError("Unaligned native year/month identifiers")
    thresholds=np.empty((12,y.shape[2],y.shape[3]),dtype=np.float32)
    counts={}
    for month in range(1,13):
        subset=y[train & (months==month),0]
        counts[str(month)]=int(len(subset))
        if len(subset)!=75:
            raise ValueError("BCC noleap P95 requires exactly 75 training targets per month")
        thresholds[month-1]=np.quantile(subset,0.95,axis=0).astype(np.float32)
    return thresholds,counts


def verify_saved_npz(path):
    with np.load(path,allow_pickle=False) as z:
        needed={"X","y","years","months","latitude","longitude_east",
                "source_native_time","target_native_time",
                "source_cf_day_value","target_cf_day_value",
                "train_mask","monthly_train_p95_K"}
        if set(z.files)!=needed:
            raise ValueError("Unexpected BCC NPZ schema or missing native timestamps")
        x,y,yr,mo=z["X"],z["y"],z["years"],z["months"]
        train=z["train_mask"].astype(bool)
        if (x.shape!=y.shape or x.shape!=(1140,1,23,23)
                or int(train.sum())!=900 or int((~train).sum())!=240):
            raise ValueError("BCC regional data shape/count mismatch")
        if not np.all(z["target_cf_day_value"]-z["source_cf_day_value"]==1):
            raise ValueError("Native CF numeric day lead must equal one")
        if any(int(np.sum(train & (mo==m)))!=75 for m in range(1,13)):
            raise ValueError("BCC historical-training month counts violated")
        if not np.all(yr[train]<=2004) or not np.all((yr[~train]>=2005)&(yr[~train]<=2014)):
            raise ValueError("Historical fold year contamination")
        if z["monthly_train_p95_K"].shape!=(12,23,23):
            raise ValueError("Local training-only P95 must match BCC grid")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite regional BCC sample")


def extract(manifest, output):
    import fsspec
    import xarray as xr

    rows=schedule()
    validate_schedule(rows)
    source=get_model(manifest)["historical"]["store"]
    if "/historical/" not in source or "/day/tas/" not in source:
        raise ValueError("Historical-only daily tas source required")
    ds=xr.open_zarr(fsspec.get_mapper(source,anon=True,token="anon"),
                    consolidated=True,decode_times=False,chunks=None,mask_and_scale=True)
    try:
        tas=ds["tas"]
        if tas.attrs.get("units")!="K" or set(tas.dims)!={"time","lat","lon"}:
            raise ValueError("Expected native daily Kelvin BCC temperature")
        latitude=np.asarray(ds["lat"].values,dtype=np.float64)
        longitude=np.asarray(ds["lon"].values,dtype=np.float64)%360
        lat_sel=contiguous_span(latitude,25.,50.)
        lon_sel=contiguous_span(longitude,235.,260.)
        lat=latitude[lat_sel]; lon=longitude[lon_sel]
        if (len(lat),len(lon))!=(23,23):
            raise ValueError("BCC native regional grid not matching audited v5.0")
        t=ds["time"]
        calendar=str(t.attrs.get("calendar",t.encoding.get("calendar","standard")))
        units=str(t.attrs.get("units",""))
        if calendar not in ("365_day","noleap") or "days since" not in units:
            raise ValueError("Require native 365_day/noleap CF daily coordinate")
        numeric=np.asarray(t.values,dtype=float)
        if not np.all(np.diff(numeric)>0):
            raise ValueError("Nonincreasing source native time")
        xs=[];ys=[];yrs=[];months=[];sdate=[];tdate=[];snum=[];tnum=[];masks=[]
        for fold,year,month,day in rows:
            i,j,origin,destination=native_noleap_pair(numeric,year,month,day,units,calendar)
            x=np.asarray(tas.isel(time=i,lat=lat_sel,lon=lon_sel).values,dtype=np.float32)
            y=np.asarray(tas.isel(time=j,lat=lat_sel,lon=lon_sel).values,dtype=np.float32)
            if x.shape!=(23,23) or y.shape!=(23,23) or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("BCC native-grid data contain missing/incompatible cells")
            if min(x.min(),y.min())<150 or max(x.max(),y.max())>350:
                raise ValueError("Kelvin physical bounds failed")
            if not np.isclose(numeric[j]-numeric[i],1.0,atol=1e-8):
                raise ValueError("Not exactly one native CF day")
            xs.append(x[None]);ys.append(y[None]);yrs.append(year);months.append(month)
            sdate.append(origin);tdate.append(destination)
            snum.append(numeric[i]);tnum.append(numeric[j]);masks.append(fold=="train")
        xx=np.stack(xs);yy=np.stack(ys)
        years=np.asarray(yrs,dtype=np.int32)
        mm=np.asarray(months,dtype=np.int8)
        train=np.asarray(masks,dtype=bool)
        origin=np.asarray(snum,dtype=np.float64)
        target=np.asarray(tnum,dtype=np.float64)
        if not target[train].max() < origin[~train].min():
            raise ValueError("Train/validation target-origin time boundary leak")
        p95,counts=monthly_training_reference(yy,years,mm,train)
        out=Path(output)
        if out.exists() or out.with_suffix(".json").exists():
            raise FileExistsError("Cannot overwrite frozen source cohort")
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out, X=xx,y=yy,years=years,months=mm,latitude=lat,
            longitude_east=lon,source_native_time=np.asarray(sdate,dtype="U24"),
            target_native_time=np.asarray(tdate,dtype="U24"),
            source_cf_day_value=origin,target_cf_day_value=target,
            train_mask=train,monthly_train_p95_K=p95)
        verify_saved_npz(out)
        weights=np.cos(np.deg2rad(lat))[:,None]
        diff=(xx[~train]-yy[~train])[:,0]
        persistence=float(np.sqrt(np.sum(weights[None,:,:]*diff**2)/
                                  (weights.sum()*len(lon)*(~train).sum())))
        report={
            "status":"v51_bcc_historical_only_noleap_native_pairs_verified_no_training",
            "source_id":"BCC-CSM2-MR","member_id":"r1i1p1f1","grid_label":"gn",
            "source":source,"model_calendar":calendar,"time_units":units,
            "source_catalog_sha256":manifest["catalog_sha256"],
            "train_years":[1980,2004],"validation_years":[2005,2014],
            "sample_source_days":{"train":list(TRAIN_DAYS),"validation":list(VALID_DAYS)},
            "samples_per_train_month":counts,
            "train_count":int(train.sum()),"validation_count":int((~train).sum()),
            "native_grid_shape":[23,23],"regional_latitude_bounds":[float(lat.min()),float(lat.max())],
            "regional_east_longitude_bounds":[float(lon.min()),float(lon.max())],
            "first_source_time_native":sdate[0],"last_target_time_native":tdate[-1],
            "historical_validation_persistence_area_rmse_K":persistence,
            "input_npz_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
            "schedule_sha256":hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
            "local_monthly_p95_reference":"training-only 75 native-calendar target days/month",
            "ssp_data_accessed":False,"model_trained":False,
            "limitations":[
                "Sparse 3/2 dates per month, not continuous daily climate data",
                "Native BCC noleap day index must not be cast to Gregorian datetime64",
                "BCC-local P95 based on historical train only; AWI scalers and checkpoints prohibited",
                "No BCC future SSP forecast skill established",
            ],
        }
        out.with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:report[k] for k in ("status","train_count","validation_count",
            "model_calendar","native_grid_shape","historical_validation_persistence_area_rmse_K",
            "input_npz_sha256")},indent=2))
        return report
    finally:
        ds.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    extract(json.loads(Path(args.manifest).read_text()),Path(args.output))


if __name__=="__main__":
    main()
