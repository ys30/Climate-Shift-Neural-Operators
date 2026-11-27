"""E1 v5.24 — AWI-CM-1-1-MR frozen Historical-only 2050 contiguous summer repeat.

Identical 91 native summer source days, +1 day Kelvin daily mean, and >=3
consecutive same-pixel historical-month-P95 event-day rule as CAMS v5.20.
All AWI Historical-only 18 checkpoint SHAs verified before SSP access.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from climate_shift.models import FNO2d
from climate_shift.unet import UNet2d
from climate_shift.metrics import latitude_weighted_metrics
from scripts.audit_cmip6_awi_daily_pairs import contiguous_span,exact_daily_pair_indices
from scripts.evaluate_cmip6_v48_frozen_ssp import manifest_entry,read_frozen_checkpoints
from climate_shift.seasonal_extremes import local_monthly_hot_diagnostics
from scripts.run_cmip6_v47_historical_arms import REQUIRED_SHA
from scripts.evaluate_cmip6_v520_cams_contiguous_summer import (
    dates,verify_dates,event_stats,summarize_seed_values)

STATUS="v524_AWI_independent_frozen_contiguous_2050_summer_daily_mean_heat_events"
SSPS=("ssp245","ssp585")
MODELS=("fno","unet")
WEIGHTS=(0,1,3)
SEEDS=(11,22,33)

def read_awi_91(store,scenario,frozen):
    import fsspec,xarray as xr
    if scenario not in SSPS or f"/{scenario}/" not in store or "/day/tas/" not in store:
        raise ValueError("AWI source experiment mismatch")
    ds=xr.open_zarr(fsspec.get_mapper(store,anon=True,token="anon"),
                    consolidated=True,decode_times=False,chunks=None,mask_and_scale=True)
    try:
        tas=ds["tas"]
        if tas.attrs.get("units")!="K" or set(tas.dims)!={"time","lat","lon"}:
            raise ValueError("AWI daily Kelvin native source required")
        lat=np.asarray(ds["lat"].values,dtype=float)
        lon=np.asarray(ds["lon"].values,dtype=float)%360
        ilat=contiguous_span(lat,25,50);ilon=contiguous_span(lon,235,260)
        if not np.array_equal(lat[ilat],frozen["latitude"]) or not np.array_equal(lon[ilon],frozen["lon"]):
            raise ValueError("AWI SSP native geometry differs from frozen Historical")
        t=ds["time"]
        cal=str(t.attrs.get("calendar",t.encoding.get("calendar","standard")))
        units=str(t.attrs.get("units",""))
        if cal!="proleptic_gregorian" or "days since" not in units:
            raise ValueError("AWI native proleptic Gregorian calendar required")
        time=np.asarray(t.values,dtype=float)
        plan=dates();verify_dates(plan)
        positions=[]
        for year,month,day in plan:
            i,j,src,dst=exact_daily_pair_indices(time,(year,month,day),units,cal)
            if src[:10]!=f"{year:04d}-{month:02d}-{day:02d}" or j!=i+1:
                raise ValueError("AWI wrong native target day")
            positions.append((i,j,dst))
        if not (np.all(np.diff([z[0] for z in positions])==1)
            and np.all(np.diff([z[1] for z in positions])==1)):
            raise ValueError("AWI summer target days not contiguous")
        start,end=positions[0][0],positions[-1][1]
        fields=np.asarray(tas.isel(time=slice(start,end+1),lat=ilat,lon=ilon).values,dtype=np.float32)
        if fields.shape!=(92,26,27) or not np.isfinite(fields).all():
            raise ValueError("Missing AWI daily Kelvin native fields")
        if fields.min()<150 or fields.max()>350:
            raise ValueError("AWI implausible regional Kelvin")
        months=np.asarray([int(dst[5:7]) for _,_,dst in positions],dtype=np.int64)
        if not np.array_equal(months,np.array([6]*29+[7]*31+[8]*31)):
            raise ValueError("Source-target summer calendar boundary wrong")
        return fields[:-1,None],fields[1:,None],months,{
          "source_store":store,"native_calendar":cal,"native_grid_shape":[26,27],
          "n_source_days":91,"n_bounded_regional_native_days_read":92,
          "first_target":positions[0][2],"last_target":positions[-1][2]}
    finally:ds.close()

def predict(frozen,name,weight,seed,X):
    import torch
    ck=frozen["models"][(name,weight,seed)][0]
    model=(FNO2d(width=8,modes_y=4,modes_x=4,depth=2)
           if name=="fno" else UNet2d(width=8))
    model.load_state_dict(ck["state_dict"],strict=True);model.eval()
    xx=(X-frozen["train_mean"])/frozen["train_std"]
    with torch.no_grad():
        out=np.concatenate([model(torch.from_numpy(b)).numpy()
             for b in np.array_split(xx,7)],axis=0)*frozen["train_std"]+frozen["train_mean"]
    if out.shape!=X.shape or not np.isfinite(out).all():
        raise ValueError("Frozen BCC SSP forecast invalid")
    return out

def score(pred,target,months,frozen):
    weighted=latitude_weighted_metrics(pred,target,frozen["latitude"])
    # AWI uses the same month-local P95 source but its original date-aware diagnostic.
    times=np.array([np.datetime64(f"2050-{int(m):02d}-15") for m in months],dtype="datetime64[ns]")
    hot=local_monthly_hot_diagnostics(pred,target,times,frozen["local_p95"],latitude=frozen["latitude"])
    event=event_stats(pred,target,months,frozen["local_p95"],frozen["latitude"])
    return {"rmse_K":float(weighted["rmse"]),"mae_K":float(weighted["mae"]),
        "bias_K":float(weighted["bias"]),"daily_p95_recall":hot["hot_recall"],
        "daily_p95_precision":hot["hot_precision"],**event}

def evaluate(frozen,manifest):
    import torch
    if len(frozen["models"])!=18:
        raise ValueError("Incomplete pre-SSP frozen AWI checkpoint gate")
    row=manifest_entry(manifest)
    torch.set_num_threads(1)
    out={"status":STATUS,"source_model":"AWI-CM-1-1-MR","source_member":"r1i1p1f1",
        "target_variable":"native_daily_mean_tas_K","year":2050,
        "source_count_per_ssp":91,
        "event_definition":"same pixel >=3 consecutive days above AWI Historical train-only monthly local tas P95",
        "historical_npz_sha256":REQUIRED_SHA,
        "ssp_fitted":False,
        "checkpoint_sha":frozen["checkpoint_sha256_by_arm"],
        "scenarios":{},
        "limitations":[
           "Single AWI model member and single summer season; exploratory replication with CAMS and BCC",
           "Native Gregorian AWI daily mean tas, not tasmax or observed human heat exposure",
           "Summer boundary events censored; event-day spatial-temporal coincidence, not object tracking",
           "2050 source-target dates continuous; no claims beyond 91 target days",
           "Fixed Historical-only P95 and normalization; no SSP model or threshold refit",
           "Scenario differences within a GCM are correlated, not independent climate models"]}
    for scenario in SSPS:
        X,Y,months,meta=read_awi_91(row[scenario]["store"],scenario,frozen)
        base=score(X,Y,months,frozen)
        arms={}
        for name in MODELS:
            arms[name]={}
            for weight in WEIGHTS:
                indiv=[score(predict(frozen,name,weight,seed,X),Y,months,frozen)
                       for seed in SEEDS]
                arms[name][str(weight)]=summarize_seed_values(indiv)
        out["scenarios"][scenario]={"n_pairs":91,"source":meta,
                                   "persistence":base,"models":arms}
    return out

def main():
    p=argparse.ArgumentParser()
    for x in ("historical","arm-dir","manifest","output"):p.add_argument("--"+x,required=True)
    a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError("Do not overwrite BCC source evidence")
    # Strictly verify all 18 BCC Historical-only checkpoints BEFORE any future SSP read.
    frozen=read_frozen_checkpoints(Path(a.arm_dir),Path(a.historical))
    r=evaluate(frozen,json.loads(Path(a.manifest).read_text()))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":r["status"],"scenarios":{s:{
        "pairs":v["n_pairs"],
        "persistence_event_day_recall":v["persistence"]["event_day_recall"],
        "fno1_event_day_recall":v["models"]["fno"]["0"]["event_day_recall"]["mean"],
        "fno1_rmse_K":v["models"]["fno"]["0"]["rmse_K"]["mean"]}
        for s,v in r["scenarios"].items()}},indent=2))
if __name__=="__main__":main()
