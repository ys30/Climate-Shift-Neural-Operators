"""E1 v5.16 CAMS native Historical-only +1-day training cohort.

Source identity and regional geometry are pinned by the independently verified
v5.15 audit. SSP stores are NEVER opened. Native cftime strings/CF numeric
day values avoid incorrectly coercing non-Gregorian calendars.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.audit_cmip6_awi_daily_pairs import contiguous_span,exact_daily_pair_indices
from scripts.export_cmip6_bcc_v51_historical import schedule,validate_schedule,monthly_training_reference

GCM="CAMS-CSM1-0"
PREVIOUS="v515_CAMS_native_western_US_regional_one_day_real_source_gate"
STATUS="v516_CAMS_native_historical_only_1140_pairs_training_ready"

def validate_gate(d):
    if d.get("status")!=PREVIOUS or d.get("model")!=GCM or d.get("no_neural_training") is not True:
        raise ValueError("CAMS v5.15 verified origin required")
    scenarios=d.get("experiments",{})
    if set(scenarios)!=set(("historical","ssp245","ssp585")):
        raise ValueError("Incomplete archived source provenance")
    actual=scenarios["historical"]
    if (actual.get("status")!="actual_bcc_native_one_day_pair_sanity_only"
            or len(actual.get("pairs",[]))!=2):
        raise ValueError("Unverified native Historical source")
    shape=tuple(d["native_regional_shape"])
    if len(shape)!=2 or min(shape)<4 or shape[0]*shape[1]>2000:
        raise ValueError("Unexpected CAMS grid")
    for rec in scenarios.values():
        if tuple(rec["grid_shape"])!=shape or rec["calendar"]!=actual["calendar"]:
            raise ValueError("Historical/SSP native geometry changed")
    url=actual["store"]
    if (not url.startswith("gs://cmip6/") or f"/{GCM}/" not in url
            or "/historical/" not in url or "/day/tas/" not in url):
        raise ValueError("Unverified Historical-only store")
    return shape,url,actual

def verify_npz(path,shape):
    with np.load(path,allow_pickle=False) as d:
        expected={"X","y","years","months","latitude","longitude_east","train_mask",
                  "source_native_time","target_native_time","source_cf_day_value",
                  "target_cf_day_value","monthly_train_p95_K"}
        if set(d.files)!=expected:
            raise ValueError("Unexpected CAMS saved schema")
        if (d["X"].shape!=d["y"].shape or d["X"].shape!=(1140,1,*shape)
                or d["monthly_train_p95_K"].shape!=(12,*shape)):
            raise ValueError("Incomplete CAMS native training grid")
        train=d["train_mask"].astype(bool)
        if train.sum()!=900 or (~train).sum()!=240:
            raise ValueError("CAMS Historical fold contamination")
        if not np.allclose(d["target_cf_day_value"]-d["source_cf_day_value"],1,atol=1e-7):
            raise ValueError("Non-native +1 day pairing")
        if not (np.isfinite(d["X"]).all() and np.isfinite(d["y"]).all()):
            raise ValueError("CAMS saved Kelvin values not finite")
        if not all(np.sum(train & (d["months"]==m))==75 for m in range(1,13)):
            raise ValueError("CAMS monthly training cohort incomplete")
        if not all(np.sum((~train) & (d["months"]==m))==20 for m in range(1,13)):
            raise ValueError("CAMS monthly validation cohort incomplete")
        if not np.all(d["years"][train]<=2004) or not np.all((d["years"][~train]>=2005)&(d["years"][~train]<=2014)):
            raise ValueError("Future or wrong year in native Historical source")

def export(verified,out):
    import fsspec
    import xarray as xr
    shape,store,pinned=validate_gate(verified)
    dates=schedule()
    validate_schedule(dates)
    out=Path(out)
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError("Cannot overwrite historical source artifact")
    ds=xr.open_zarr(fsspec.get_mapper(store,anon=True,token="anon"),
                    consolidated=True,decode_times=False,chunks=None,mask_and_scale=True)
    try:
        tas=ds["tas"]
        if tas.attrs.get("units")!="K" or set(tas.dims)!={"time","lat","lon"}:
            raise ValueError("Native daily tas Kelvin 3D grid required")
        lat_full=np.asarray(ds["lat"].values,dtype=np.float64)
        lon_full=np.asarray(ds["lon"].values,dtype=np.float64)%360
        ilat=contiguous_span(lat_full,25,50)
        ilon=contiguous_span(lon_full,235,260)
        lat=lat_full[ilat].copy()
        lon=lon_full[ilon].copy()
        if (len(lat),len(lon))!=shape:
            raise ValueError("CAMS grid differs from pinned v5.15")
        if (hashlib.sha256(lat.tobytes()).hexdigest()!=pinned["latitude_sha256"]
             or hashlib.sha256(lon.tobytes()).hexdigest()!=pinned["longitude_sha256"]):
            raise ValueError("Native coordinate hashes changed from v5.15")
        t=ds["time"]
        cal=str(t.attrs.get("calendar",t.encoding.get("calendar","standard")))
        units=str(t.attrs.get("units",""))
        if cal!=pinned["calendar"] or "days since" not in units:
            raise ValueError("Frozen CAMS native calendar changed")
        native=np.asarray(t.values,dtype=float)
        if native.ndim!=1 or not np.all(np.diff(native)>0):
            raise ValueError("Nonmonotonic Historical CF coordinates")
        xs=[];ys=[];years=[];months=[];train=[];src_dates=[];dst_dates=[];ti=[];tj=[]
        for fold,year,month,day in dates:
            i,j,origin,target=exact_daily_pair_indices(native,(year,month,day),units,cal)
            if origin[:10]!=f"{year:04}-{month:02}-{day:02}" or target[:4]!=str(year):
                raise ValueError("Incorrect native CAMS +1 calendar day")
            x=np.asarray(tas.isel(time=i,lat=ilat,lon=ilon).values,dtype=np.float32)
            y=np.asarray(tas.isel(time=j,lat=ilat,lon=ilon).values,dtype=np.float32)
            if x.shape!=shape or y.shape!=shape or not(np.isfinite(x).all() and np.isfinite(y).all()):
                raise ValueError("Regional Historical missing or invalid values")
            if min(x.min(),y.min())<150 or max(x.max(),y.max())>350:
                raise ValueError("Physically implausible Kelvin data")
            if not np.isclose(native[j]-native[i],1.,atol=1e-8):
                raise ValueError("Native +1 day CF interval changed")
            xs.append(x[None]);ys.append(y[None])
            years.append(year);months.append(month);train.append(fold=="train")
            src_dates.append(origin);dst_dates.append(target)
            ti.append(native[i]);tj.append(native[j])
        X=np.stack(xs)
        Y=np.stack(ys)
        year_arr=np.asarray(years,dtype=np.int32)
        month_arr=np.asarray(months,dtype=np.int8)
        mask=np.asarray(train,dtype=bool)
        ti=np.asarray(ti,dtype=np.float64);tj=np.asarray(tj,dtype=np.float64)
        if not (X.shape==Y.shape==(1140,1,*shape) and mask.sum()==900
                and (~mask).sum()==240 and tj[mask].max()<ti[~mask].min()):
            raise ValueError("Historical fold leak or missing samples")
        p95,counts=monthly_training_reference(Y,year_arr,month_arr,mask)
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,X=X,y=Y,years=year_arr,months=month_arr,
            latitude=lat,longitude_east=lon,train_mask=mask,
            source_native_time=np.asarray(src_dates,dtype="U32"),
            target_native_time=np.asarray(dst_dates,dtype="U32"),
            source_cf_day_value=ti,target_cf_day_value=tj,
            monthly_train_p95_K=p95)
        verify_npz(out,shape)
        w=np.cos(np.deg2rad(lat))[:,None]
        delta=(X[~mask]-Y[~mask])[:,0]
        rmse=float(np.sqrt(np.sum(w[None,:,:]*delta**2)/(w.sum()*len(lon)*(~mask).sum())))
        report={
          "status":STATUS,"source_id":GCM,
          "member_id":verified["member"],"grid_label":verified["grid"],
          "source_store":store,"catalog_sha256":verified["catalog_sha256"],
          "native_calendar":cal,"time_units":units,
          "native_grid_shape":list(shape),"train_years":[1980,2004],
          "validation_years":[2005,2014],
          "source_days_train":[6,16,26],"source_days_validation":[8,20],
          "train_pairs":900,"validation_pairs":240,
          "training_targets_per_month":counts,
          "train_only_monthly_p95":True,
          "independent_CAMS_scalers_required":True,
          "ssp_data_opened":False,"model_trained":False,
          "historical_validation_persistence_rmse_K":rmse,
          "npz_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
          "limitations":["Sparse native calendar dates, not full 1980–2014 daily time series",
                         "CAMS Historical validation only; no CAMS future SSP forecasting yet",
                         "Validation cohort must remain isolated from model fitting",
                         "No pooling with AWI/BCC native model normalizers or thresholds"]
        }
        out.with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        return report
    finally:
        ds.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--v515",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    prior=json.loads(Path(args.v515).read_text())
    report=export(prior,Path(args.output))
    print(json.dumps({k:report[k] for k in ("status","native_calendar","native_grid_shape",
            "train_pairs","validation_pairs","npz_sha256")},indent=2))
if __name__=="__main__":
    main()
