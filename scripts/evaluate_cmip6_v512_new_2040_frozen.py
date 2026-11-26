"""E1 v5.12: frozen FNO/U-Net on the NEW v5.11 2040 sample dates.

Independent of old SSP 2030/2050/2080 scoring dates. All historical data
and checkpoint hashes are verified before opening any future projections.
Never fit or select on this future cohort.
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
from scripts.evaluate_cmip6_v48_frozen_ssp import read_frozen_checkpoints,manifest_entry
from scripts.evaluate_cmip6_bcc_v54_frozen_ssp import inspect_frozen_sources
from scripts.audit_cmip6_awi_daily_pairs import contiguous_span,exact_daily_pair_indices
from scripts.audit_cmip6_bcc_regional_pairs import get_model
from scripts.audit_cmip6_v511_new_2040_cohort import schedule,validate_records
from scripts.fit_cmip6_bcc_v52_historical import monthly_local_hot
from climate_shift.seasonal_extremes import local_monthly_hot_diagnostics

GCM=("AWI","BCC")
SSP=("ssp245","ssp585")
MODELS=("fno","unet")
WEIGHTS=(0,1,3)
SEEDS=(11,22,33)
SOURCES={
    "AWI":{"hist":"ac0a354032836c5de5463b122d26fad2270fd5ba2ebb48f43074105c8ce7b914",
           "shape":(26,27),"calendar":"proleptic_gregorian"},
    "BCC":{"hist":"a23260e8f14a23fb57483038bcd6ca03ba74c7944a48ba29d62bcc5bd354f452",
           "shape":(23,23),"calendar":"noleap"}
}

def verify_pilot(path):
    d=json.loads(Path(path).read_text())
    if d.get("status")!="v511_verified_new_2040_seven_dates_per_month_source_cohort" or d.get("source_only") is not True or d.get("year")!=2040 or d.get("days_per_month")!=[3,7,11,15,19,23,27]:
        raise ValueError("Unverified v5.11 2040 source audit")
    validate_records(d["rows"])
    return {(r["gcm"],r["scenario"]):r for r in d["rows"]}

def verify_frozen(awi_npz,bcc_npz,awi_arms,bcc_arms):
    awi=read_frozen_checkpoints(Path(awi_arms),Path(awi_npz))
    bcc=inspect_frozen_sources(bcc_npz,bcc_arms)
    if len(awi["models"])!=18 or len(bcc["checkpoints"])!=18:
        raise ValueError("Missing historical frozen checkpoints")
    return {"AWI":awi,"BCC":bcc}

def native_arrays(store,gcm,ssp,pilot):
    import fsspec
    import xarray as xr
    if gcm not in GCM or ssp not in SSP or not store.startswith("gs://cmip6/") or f"/{ssp}/" not in store or "/day/tas/" not in store:
        raise ValueError("Unapproved projected source")
    ds=xr.open_zarr(fsspec.get_mapper(store,anon=True,token="anon"),
                    consolidated=True,decode_times=False,chunks=None,mask_and_scale=True)
    try:
        tas=ds["tas"]
        if tas.attrs.get("units")!="K" or set(tas.dims)!={"time","lat","lon"}:
            raise ValueError("Unexpected Kelvin target dimensions")
        lat=np.asarray(ds["lat"].values,dtype=np.float64)
        lon=np.asarray(ds["lon"].values,dtype=np.float64)%360
        lati=contiguous_span(lat,25.,50.)
        loni=contiguous_span(lon,235.,260.)
        rl=lat[lati];ro=lon[loni]
        if (len(rl),len(ro))!=SOURCES[gcm]["shape"]:
            raise ValueError("Native GCM grid changed")
        if hashlib.sha256(rl.tobytes()).hexdigest()!=pilot["latitude_sha256"] or hashlib.sha256(ro.tobytes()).hexdigest()!=pilot["longitude_sha256"]:
            raise ValueError("GCM coordinates do not match independently archived pilot")
        time=ds["time"]
        cal=str(time.attrs.get("calendar",time.encoding.get("calendar","standard")))
        units=str(time.attrs.get("units",""))
        if (cal!=SOURCES[gcm]["calendar"] and not(gcm=="BCC" and cal=="365_day")) or "days since" not in units:
            raise ValueError("Changed native calendar")
        numeric=np.asarray(time.values,dtype=float)
        pairs=[]
        for year,month,day in schedule():
            i,j,src,dst=exact_daily_pair_indices(numeric,(year,month,day),units,cal)
            if src[:10]!=f"{year:04d}-{month:02d}-{day:02d}" or dst[:4]!=str(year):
                raise ValueError("Wrong native +1 day source/target")
            pairs.append((i,j))
        xs=[];ys=[]
        for start in range(0,84,14):
            portion=pairs[start:start+14]
            x=np.asarray(tas.isel(time=[p[0] for p in portion],lat=lati,lon=loni).values,dtype=np.float32)
            y=np.asarray(tas.isel(time=[p[1] for p in portion],lat=lati,lon=loni).values,dtype=np.float32)
            if x.shape!=(len(portion),*SOURCES[gcm]["shape"]) or y.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Missing source/target grid data")
            xs.append(x);ys.append(y)
        x=np.concatenate(xs);y=np.concatenate(ys)
        if x.shape[0]!=84 or y.shape!=x.shape:
            raise ValueError("Incomplete 2040 date cohort")
        # Independently archived v5.11 source means act as guards on exact day ordering.
        from scripts.audit_cmip6_v59_source_warming import weighted_mean
        for index,record in enumerate(pilot["daily_records"]):
            if abs(weighted_mean(x[index],rl)-record["source_regional_K"])>1e-4 or abs(weighted_mean(y[index],rl)-record["target_regional_K"])>1e-4:
                raise ValueError("Re-read projected fields disagree with frozen v5.11 source audit")
        return x[:,None],y[:,None],rl
    finally:
        ds.close()

def metrics(pred,target,lat,p95,gcm):
    dates=np.array([np.datetime64(f"2040-{m:02d}-{d+1:02d}T12:00:00") for _,m,d in schedule()],dtype="datetime64[ns]")
    months=np.array([m for _,m,_ in schedule()])
    if gcm=="AWI":
        tail=local_monthly_hot_diagnostics(pred,target,dates,p95,latitude=lat)
    else:
        tail=monthly_local_hot(pred,target,months,p95,lat)
    weighted=latitude_weighted_metrics(pred,target,lat)
    return {
        "rmse_K":float(weighted["rmse"]),
        "mae_K":float(weighted["mae"]),
        "bias_K":float(weighted["bias"]),
        "p95_recall":float(tail["hot_recall"]),
        "p95_precision":float(tail["hot_precision"]),
        "predicted_hot_fraction":float(tail["predicted_hot_fraction"])
    }

def score_model(gcm,frozen,x):
    import torch
    if gcm=="AWI":
        get=lambda m,w,s:frozen["models"][(m,w,s)][0]
        mu=frozen["train_mean"];sd=frozen["train_std"]
    else:
        get=lambda m,w,s:frozen["checkpoints"][(m,w,s)]["checkpoint"]
        mu=frozen["mean"];sd=frozen["sd"]
    if not np.isfinite(sd) or sd<=0:
        raise ValueError("Invalid Historical-only standard deviation")
    normalized=(x-mu)/sd
    results={}
    for m in MODELS:
        for w in WEIGHTS:
            for seed in SEEDS:
                ck=get(m,w,seed)
                model=FNO2d(width=8,modes_y=4,modes_x=4,depth=2) if m=="fno" else UNet2d(width=8)
                model.load_state_dict(ck["state_dict"],strict=True)
                model.eval()
                with torch.no_grad():
                    pred=np.concatenate([model(torch.from_numpy(b)).numpy()
                        for b in np.array_split(normalized,6)]).astype(np.float32)*sd+mu
                if pred.shape!=x.shape or not np.isfinite(pred).all():
                    raise ValueError("Invalid frozen model prediction")
                results[m,w,seed]=pred
    if len(results)!=18:raise ValueError("Not all historical checkpoints evaluated")
    return results

def run(awi_hist,bcc_hist,awi_arms,bcc_arms,awi_manifest,bcc_manifest,pilot_json):
    pilot=verify_pilot(pilot_json)
    frozen=verify_frozen(awi_hist,bcc_hist,awi_arms,bcc_arms)
    awi=manifest_entry(json.loads(Path(awi_manifest).read_text()))
    bcc=get_model(json.loads(Path(bcc_manifest).read_text()))
    records=[]
    for g,model_row in (("AWI",awi),("BCC",bcc)):
        ref=frozen[g]
        lat_ref=ref["latitude"] if g=="AWI" else ref["lat"]
        p95=ref["local_p95"] if g=="AWI" else ref["p95"]
        for ssp in SSP:
            x,y,lat=native_arrays(model_row[ssp]["store"],g,ssp,pilot[g,ssp])
            if not np.allclose(lat,lat_ref,atol=1e-8):
                raise ValueError("Historical-vs-future native latitude mismatch")
            results={"persistence":metrics(x,y,lat,p95,g)}
            predictions=score_model(g,ref,x)
            for m in MODELS:
                for w in WEIGHTS:
                    scores=[metrics(predictions[m,w,seed],y,lat,p95,g) for seed in SEEDS]
                    results[f"{m}_weight{w}"]={
                        metric:{"individual":[v[metric] for v in scores],
                                "mean":float(np.mean([v[metric] for v in scores])),
                                "sd_across_seeds":float(np.std([v[metric] for v in scores],ddof=1))}
                        for metric in scores[0]
                    }
            records.append({"gcm":g,"ssp":ssp,"year":2040,"n_pairs":84,"models":results,
                "native_grid_shape":list(SOURCES[g]["shape"]),"calendar":SOURCES[g]["calendar"]})
    if len(records)!=4:raise ValueError("Missing GCM × SSP future evaluation")
    return {"status":"v512_frozen_historical_neural_forecasts_new_2040_cohort",
        "frozen_training":True,"checkpoint_selection_using_2040":False,
        "date_plan":[3,7,11,15,19,23,27],"year":2040,"rows":records,
        "limitations":[
            "84 sparsely selected 2040 forecast pairs per GCM/SSP, not full-year heatwaves or forced warming",
            "One CMIP6 member per GCM; projected scenario targets are simulated rather than observed",
            "Previously explored GCMs/SSP scenarios: new date cohort is supplementary rather than fully blind research design",
            "No model or loss-weight selection, SSP fitting or checkpoint adaptation"
        ]}

def main():
    p=argparse.ArgumentParser()
    for s in ("awi-hist","bcc-hist","awi-arms","bcc-arms","awi-manifest","bcc-manifest","pilot-json","output"):
        p.add_argument("--"+s,required=True)
    a=vars(p.parse_args());out=Path(a["output"])
    if out.exists():raise FileExistsError("Immutable experiment cannot overwrite")
    d=run(a["awi_hist"],a["bcc_hist"],a["awi_arms"],a["bcc_arms"],
          a["awi_manifest"],a["bcc_manifest"],a["pilot_json"])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":d["status"],"rows":len(d["rows"])},indent=2))
if __name__=="__main__":main()
