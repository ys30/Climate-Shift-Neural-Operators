"""E1 v5.20 — CAMS 2050 native contiguous summer heat-event pilot.

Source dates: 2050-06-01 .. 2050-08-30 (91 consecutive native days).
Prediction target: next daily mean tas, June 2 .. August 31.
Heat threshold is CAMS HISTORICAL train-only monthly-local TAS P95.
Detection events are >=3 CONSECUTIVE hot daily-mean TAS targets per pixel.
This is *not* a heatwave analysis of tasmax, observed weather, or a full year.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from climate_shift.models import FNO2d
from climate_shift.unet import UNet2d
from climate_shift.metrics import latitude_weighted_metrics
from scripts.audit_cmip6_awi_daily_pairs import contiguous_span, exact_daily_pair_indices
from scripts.evaluate_cmip6_v518_cams_frozen_ssp import (
    inspect_frozen, manifest_for_CAMS, SSPS, MODELS, WEIGHTS, SEEDS)
from scripts.audit_cmip6_catalog import read_catalog, select_pairs
from scripts.fit_cmip6_bcc_v52_historical import monthly_local_hot

STATUS="v520_CAMS_frozen_contiguous_2050_summer_daily_mean_heat_event_pilot"
YEAR=2050
MONTHS=((6,30),(7,31),(8,30)) # August 31 source omitted; last target = August 31
RUN_MIN=3

def dates():
    return [(YEAR,month,d) for month,last in MONTHS for d in range(1,last+1)]

def verify_dates(rows):
    if len(rows)!=91 or len(set(rows))!=91:
        raise ValueError("Pilot must have 91 distinct consecutive summer source dates")
    import cftime
    # Calendar chosen by true source, 2050 is non-leap in both noleap and Gregorian calendars.
    for cal in ("noleap","proleptic_gregorian"):
        n=[cftime.date2num(cftime.datetime(*d,12,calendar=cal),
                           units="days since 2000-01-01",calendar=cal) for d in rows]
        if not np.allclose(np.diff(n),1,atol=1e-9):
            raise ValueError("Discontinuous heat-event dates")

def run_mask(binary,min_length=RUN_MIN):
    """True for all days belonging to >=min_length consecutive hot days at a pixel."""
    z=np.asarray(binary,dtype=bool)
    if z.ndim!=3 or not 2<=min_length<=z.shape[0]:
        raise ValueError("Require time×lat×lon with valid minimum duration")
    # Window Boolean intersection then paint the full event onto each of its days.
    starts=z[:len(z)-min_length+1].copy()
    for shift in range(1,min_length):
        starts &= z[shift:shift+len(starts)]
    marked=np.zeros_like(z,dtype=bool)
    for shift in range(min_length):
        marked[shift:shift+len(starts)] |= starts
    return marked

def event_stats(pred,target,months,p95,lat):
    """Area-weighted >=3-day true/predicted heat-event-day precision/recall."""
    if pred.shape!=target.shape or pred.ndim!=4 or pred.shape[1]!=1:
        raise ValueError("Mismatched daily Kelvin forecast shapes")
    if len(pred)!=91 or np.asarray(months).shape!=(91,):
        raise ValueError("Noncontinuous target-day cohort")
    if p95.shape!=(12,pred.shape[2],pred.shape[3]):
        raise ValueError("Historical source monthly threshold grid mismatched")
    obs=(target[:,0]>p95[np.asarray(months,dtype=int)-1])
    forecast=(pred[:,0]>p95[np.asarray(months,dtype=int)-1])
    true_event=run_mask(obs)
    pred_event=run_mask(forecast)
    w=np.broadcast_to(np.cos(np.deg2rad(lat))[None,:,None],true_event.shape)
    observed=float(np.sum(w*true_event))
    predicted=float(np.sum(w*pred_event))
    matched=float(np.sum(w*(true_event&pred_event)))
    # Null means not defined; never manufacture 1.0 for no observed hot events.
    return {
      "true_event_day_fraction":observed/float(np.sum(w)),
      "forecast_event_day_fraction":predicted/float(np.sum(w)),
      "event_day_recall":matched/observed if observed>0 else None,
      "event_day_precision":matched/predicted if predicted>0 else None,
      "observed_3day_pixel_windows":int(np.sum(
         obs[:-2]&obs[1:-1]&obs[2:])),
      "forecast_3day_pixel_windows":int(np.sum(
         forecast[:-2]&forecast[1:-1]&forecast[2:])),
      "boundary_censoring":"first and last forecast days may belong to events outside this summer window",
      "reference":"CAMS historical-training local monthly tas P95, not future fitted",
    }

def real_future_91(store,scenario,frozen):
    import fsspec
    import xarray as xr
    if scenario not in SSPS or f"/{scenario}/" not in store:
        raise ValueError("Wrong CAMS future scenario source")
    ds=xr.open_zarr(fsspec.get_mapper(store,anon=True,token="anon"),
                    consolidated=True,decode_times=False,chunks=None,mask_and_scale=True)
    try:
        tas=ds["tas"]
        if tas.attrs.get("units")!="K" or set(tas.dims)!={"time","lat","lon"}:
            raise ValueError("Native daily tas in Kelvin required")
        lat=np.asarray(ds["lat"].values,dtype=float)
        lon=np.asarray(ds["lon"].values,dtype=float)%360
        ilat=contiguous_span(lat,25,50);ilon=contiguous_span(lon,235,260)
        if (not np.array_equal(lat[ilat],frozen["lat"])
            or not np.array_equal(lon[ilon],frozen["lon"])):
            raise ValueError("Native CAMS summer coordinates differ from Historical")
        time=ds["time"]
        cal=str(time.attrs.get("calendar",time.encoding.get("calendar","standard")))
        unit=str(time.attrs.get("units",""))
        if cal!=frozen["calendar"] or "days since" not in unit:
            raise ValueError("CAMS native calendar changed")
        numeric=np.asarray(time.values,dtype=float)
        rows=dates()
        verify_dates(rows)
        indices=[]
        for yr,month,day in rows:
            i,j,source,target=exact_daily_pair_indices(numeric,(yr,month,day),unit,cal)
            if source[:10]!=f"{yr:04d}-{month:02d}-{day:02d}":
                raise ValueError("Changed native source schedule")
            indices.append((i,j,target))
        if not np.all(np.diff([i for i,j,_ in indices])==1):
            raise ValueError("Native CMIP6 summer source sequence not consecutive")
        if not np.all(np.diff([j for i,j,_ in indices])==1):
            raise ValueError("Native target sequence not consecutive")
        if any(j!=i+1 for i,j,_ in indices):
            raise ValueError("Temperature target is not exactly +1 native day")
        source_i=indices[0][0]
        last_j=indices[-1][1]
        # One bounded regional + contiguous time read per scenario, not a global model field.
        region=np.asarray(tas.isel(time=slice(source_i,last_j+1),
                                   lat=ilat,lon=ilon).values,dtype=np.float32)
        if region.shape!=(92,*frozen["shape"]):
            raise ValueError("Missing continuous summer target region")
        if not np.isfinite(region).all() or region.min()<150 or region.max()>350:
            raise ValueError("Missing or implausible native Kelvin in summer")
        X=region[:-1,None]
        Y=region[1:,None]
        # Source month differs at June/July boundary; derive threshold month from target dates.
        target_months=np.asarray([int(dst[5:7]) for _,_,dst in indices],dtype=np.int64)
        if len(X)!=91 or len(Y)!=91 or not np.array_equal(
            target_months,np.array([6]*29+[7]*31+[8]*31)):
            raise ValueError("Target month or consecutive source-target misalignment")
        return X,Y,target_months,{
          "source_first":str(rows[0]),"source_last":str(rows[-1]),
          "target_first":indices[0][2],"target_last":indices[-1][2],
          "source_store":store,"calendar":cal,"grid_shape":list(frozen["shape"]),
          "n_source_days":91,"n_bounded_regional_native_days_read":92
        }
    finally:
        ds.close()

def fixed_prediction(frozen,model_name,weight,seed,X):
    import torch
    ck=frozen["checkpoints"][(model_name,weight,seed)]["data"]
    model=(FNO2d(width=8,modes_y=4,modes_x=4,depth=2)
           if model_name=="fno" else UNet2d(width=8))
    model.load_state_dict(ck["state_dict"],strict=True)
    model.eval()
    mu=frozen["mu"];sd=frozen["sd"]
    normalized=(X-mu)/sd
    with torch.no_grad():
        pred=np.concatenate([model(torch.from_numpy(b)).numpy()
            for b in np.array_split(normalized,7)],axis=0)*sd+mu
    if pred.shape!=X.shape or not np.isfinite(pred).all():
        raise ValueError("Incorrect frozen CAMS summer predictions")
    return pred

def metrics(pred,Y,months,frozen):
    spatial=latitude_weighted_metrics(pred,Y,frozen["lat"])
    hot=monthly_local_hot(pred,Y,months,frozen["p95"],frozen["lat"])
    event=event_stats(pred,Y,months,frozen["p95"],frozen["lat"])
    return {"rmse_K":float(spatial["rmse"]),
            "mae_K":float(spatial["mae"]),
            "bias_K":float(spatial["bias"]),
            "daily_p95_recall":hot["hot_recall"],
            "daily_p95_precision":hot["hot_precision"],
            **event}

def summarize_seed_values(entries):
    if len(entries)!=3:raise ValueError("Require all 3 frozen Historical seeds")
    output={}
    for field in ("rmse_K","mae_K","bias_K",
                  "daily_p95_recall","daily_p95_precision",
                  "event_day_recall","event_day_precision"):
        v=[e[field] for e in entries]
        if all(x is None for x in v):
            output[field]={"individual":v,"mean":None,"sd_seeds":None}
        elif any(x is None or not np.isfinite(x) for x in v):
            raise ValueError("Missing or nonfinite seed metric: "+field)
        else:
            output[field]={"individual":list(map(float,v)),
                            "mean":float(np.mean(v)),
                            "sd_seeds":float(np.std(v,ddof=1))}
    return output

def evaluate(frozen,manifest):
    import torch
    row=manifest_for_CAMS(manifest,frozen)
    torch.set_num_threads(1)
    out={"status":STATUS,"source_model":"CAMS-CSM1-0",
        "target_variable":"native_daily_mean_tas_K",
        "year":YEAR,"target_months":[6,7,8],
        "source_count_per_ssp":91,
        "event_definition":"at least three consecutive daily mean tas targets above own historical local monthly P95",
        "ssp_fitted":False,
        "historical_source_sha256":frozen["historical_sha"],
        "checkpoint_sha":{f"{name}_weight{w}_seed{s}":v["sha256"]
                 for (name,w,s),v in frozen["checkpoints"].items()},
        "scenarios":{},
        "limits":[
          "One 2050 contiguous June-August sampled calendar window, not multiple years",
          "Event-day recall compares same pixel AND same day, not objectively tracked heatwave systems",
          "Window-edge events are censored, and forecast target June 1 is not sampled",
          "Monthly local P95 is from Historical training only; no SSP thresholds fitted",
          "GCM simulated daily mean tas is not tasmax or observed heatwave exposure",
          "Only one CAMS climate-model member; cannot infer climate trend or statistics",
          "This new continuous-date pilot was designed after earlier SSP results; exploratory"
        ]}
    for scenario in SSPS:
        X,Y,months,meta=real_future_91(row[scenario]["store"],scenario,frozen)
        baseline=metrics(X,Y,months,frozen)
        arms={}
        for model_name in MODELS:
            arms[model_name]={}
            for weight in WEIGHTS:
                scores=[metrics(fixed_prediction(frozen,model_name,weight,seed,X),
                                Y,months,frozen) for seed in SEEDS]
                arms[model_name][str(weight)]=summarize_seed_values(scores)
        out["scenarios"][scenario]={
            "source":meta,"n_pairs":91,"persistence":baseline,"models":arms}
    return out

def main():
    p=argparse.ArgumentParser()
    for k in ("historical","metadata","arm-dir","output"):
        p.add_argument("--"+k,required=True)
    a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError("Do not overwrite existing summer pilot evidence")
    frozen=inspect_frozen(a.historical,a.metadata,a.arm_dir)
    if len(frozen["checkpoints"])!=18:raise ValueError("Incomplete frozen source")
    catalog,sha=read_catalog()
    manifest=select_pairs(catalog,sha,max_models=12)
    result=evaluate(frozen,manifest)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],
      "scenarios":{k:{"n":v["n_pairs"],
                       "persistence_event_day_recall":v["persistence"]["event_day_recall"],
                       "fno_standard_rmse_K":v["models"]["fno"]["0"]["rmse_K"]["mean"],
                       "fno_standard_event_day_recall":v["models"]["fno"]["0"]["event_day_recall"]["mean"]}
                   for k,v in result["scenarios"].items()}},indent=2))
if __name__=="__main__":main()
