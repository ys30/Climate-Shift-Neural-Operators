"""E1 v5.18: blind CAMS native daily SSP forecast from 18 Historical-only checkpoints.

Strictly audit all archived checkpoint hashes and the historical-only cohort
BEFORE opening any SSP. CAMS trained on its own native GCM grid/calibration.
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
from scripts.audit_cmip6_awi_daily_pairs import contiguous_span,exact_daily_pair_indices
from scripts.audit_cmip6_catalog import read_catalog,select_pairs
from scripts.export_cmip6_v516_cams_historical import verify_npz,STATUS as HIST_STATUS
from scripts.run_cmip6_v517_cams_historical_arms import STATUS as ARM_STATUS,SEEDS,WEIGHTS,EPOCHS
from scripts.fit_cmip6_bcc_v52_historical import monthly_local_hot
from scripts.export_cmip6_bcc_v51_historical import monthly_training_reference

YEARS=(2030,2050,2080)
DAYS=(8,20)
SSPS=("ssp245","ssp585")
MODELS=("fno","unet")
OUTPUT_STATUS="v518_CAMS_Historical_frozen_sparse_SSP_predictive_skill"

def dates():
    return [(yr,mo,d) for yr in YEARS for mo in range(1,13) for d in DAYS]

def inspect_frozen(npz,metadata,arms):
    import torch
    path=Path(npz);meta=json.loads(Path(metadata).read_text())
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get("status")!=HIST_STATUS or meta.get("npz_sha256")!=sha or meta.get("ssp_data_opened") is not False:
        raise ValueError("Unverified CAMS Historical-only source")
    shape=tuple(meta["native_grid_shape"])
    verify_npz(path,shape)
    with np.load(path,allow_pickle=False) as data:
        y=data["y"].astype(np.float32)
        months=data["months"].astype(int);years=data["years"].astype(int)
        train=data["train_mask"].astype(bool)
        lat=data["latitude"].astype(float)
        lon=data["longitude_east"].astype(float)
        p95=data["monthly_train_p95_K"].astype(np.float32)
    expected,counts=monthly_training_reference(y,years,months,train)
    if (train.sum()!=900 or (~train).sum()!=240
            or not np.allclose(p95,expected,atol=5e-5,rtol=0)
            or set(counts.values())!={75}):
        raise ValueError("Historical CAMS training threshold contamination")
    mu=float(y[train].mean());sd=float(y[train].std())
    if sd<=1e-6:raise ValueError("Invalid frozen CAMS historical normalizer")
    checkpoints={}
    for weight in WEIGHTS:
        arm=Path(arms)/f"weight_{weight}"
        summ=json.loads((arm/"cams_arm_summary.json").read_text())
        if (summ.get("status")!=ARM_STATUS or summ.get("gcm")!="CAMS-CSM1-0"
            or summ.get("historical_sha256")!=sha
            or summ.get("native_shape")!=list(shape)
            or summ.get("native_calendar")!=meta["native_calendar"]
            or summ.get("tail_weight")!=weight
            or summ.get("epochs")!=EPOCHS
            or summ.get("seeds")!=list(SEEDS)
            or summ.get("ssp_fields_read") is not False
            or summ.get("future_checkpoint_selection") is not False
            or abs(summ["normalizer_train_mean_K"]-mu)>1e-5
            or abs(summ["normalizer_train_std_K"]-sd)>1e-5):
            raise ValueError("CAMS arm archive incompatible with historical source")
        for m in MODELS:
            entries=summ["models"][m]["checkpoints"]
            if len(entries)!=3 or {e["seed"] for e in entries}!=set(SEEDS):
                raise ValueError("Incomplete CAMS seed inventory")
            for e in entries:
                key=(m,weight,e["seed"])
                f=arm/e["checkpoint"]
                if key in checkpoints or hashlib.sha256(f.read_bytes()).hexdigest()!=e["sha256"]:
                    raise ValueError("Missing or modified CAMS checkpoint SHA")
                ck=torch.load(f,map_location="cpu",weights_only=True)
                if (ck["model"]!=m or ck["seed"]!=e["seed"]
                        or ck["tail_weight"]!=weight
                        or ck["historical_npz_sha256"]!=sha
                        or ck["native_grid_shape"]!=list(shape)
                        or ck["calendar"]!=meta["native_calendar"]
                        or ck["epochs"]!=EPOCHS
                        or ck["width"]!=8 or ck["fno_modes"]!=4
                        or abs(ck["normalization_mean_K"]-mu)>1e-5
                        or abs(ck["normalization_sd_K"]-sd)>1e-5
                        or not np.allclose(ck["training_only_monthly_p95_K"].numpy(),
                                            p95,rtol=0,atol=5e-5)):
                    raise ValueError("Checkpoint metadata does not match CAMS Historical-only fit")
                checkpoints[key]={"data":ck,"sha256":e["sha256"]}
    if len(checkpoints)!=18:
        raise ValueError("All 18 CAMS models required before opening SSP")
    return {"checkpoints":checkpoints,"historical_sha":sha,
            "lat":lat,"lon":lon,"p95":p95,"mu":mu,"sd":sd,
            "shape":shape,"calendar":meta["native_calendar"],
            "catalog_sha":meta["catalog_sha256"],
            "member":meta["member_id"],"grid":meta["grid_label"]}

def manifest_for_CAMS(manifest,frozen):
    if manifest.get("status")!="catalog_metadata_only_no_climate_fields_or_transfer_skill":
        raise ValueError("Unverified CMIP6 catalog")
    if manifest["catalog_sha256"]!=frozen["catalog_sha"]:
        raise ValueError("Catalog changed after historical cohort freeze")
    items=[x for x in manifest.get("selected_candidates",[])
           if (x["source_id"],x["member_id"],x["grid_label"])==
              ("CAMS-CSM1-0",frozen["member"],frozen["grid"])]
    if len(items)!=1:
        raise ValueError("Missing unique original CAMS GCM/member/grid")
    row=items[0]
    for s in SSPS:
        store=row[s]["store"]
        if (not store.startswith("gs://cmip6/") or f"/{s}/" not in store
            or "/day/tas/" not in store or "/CAMS-CSM1-0/" not in store):
            raise ValueError("Wrong future CAMS SSP source")
    return row

def read_native_future(store,scenario,frozen):
    import fsspec
    import xarray as xr
    if scenario not in SSPS or f"/{scenario}/" not in store:
        raise ValueError("Unexpected SSP source")
    ds=xr.open_zarr(fsspec.get_mapper(store,anon=True,token="anon"),
                    consolidated=True,decode_times=False,chunks=None,mask_and_scale=True)
    try:
        tas=ds["tas"]
        if tas.attrs.get("units")!="K" or set(tas.dims)!={"time","lat","lon"}:
            raise ValueError("Unexpected CAMS native daily Kelvin field")
        lat=np.asarray(ds["lat"].values,dtype=float)
        lon=np.asarray(ds["lon"].values,dtype=float)%360
        ilat=contiguous_span(lat,25,50);ilon=contiguous_span(lon,235,260)
        if (not np.array_equal(lat[ilat],frozen["lat"])
            or not np.array_equal(lon[ilon],frozen["lon"])):
            raise ValueError("SSP CAMS grid differs from Historical training")
        t=ds["time"]
        cal=str(t.attrs.get("calendar",t.encoding.get("calendar","standard")))
        units=str(t.attrs.get("units",""))
        if cal!=frozen["calendar"] or "days since" not in units:
            raise ValueError("Native CAMS calendar differs from frozen Historical training")
        nums=np.asarray(t.values,dtype=float)
        xs=[];ys=[];years=[];months=[]
        for yr,month,day in dates():
            i,j,src,dst=exact_daily_pair_indices(nums,(yr,month,day),units,cal)
            if src[:10]!=f"{yr:04}-{month:02}-{day:02}" or dst[:4]!=str(yr):
                raise ValueError("Mismatched native +1 day forecast")
            x=np.asarray(tas.isel(time=i,lat=ilat,lon=ilon).values,dtype=np.float32)
            y=np.asarray(tas.isel(time=j,lat=ilat,lon=ilon).values,dtype=np.float32)
            if x.shape!=frozen["shape"] or y.shape!=frozen["shape"] or not (
                np.isfinite(x).all() and np.isfinite(y).all()):
                raise ValueError("Native CAMS forecast array mismatch or missingness")
            if min(x.min(),y.min())<150 or max(x.max(),y.max())>350:
                raise ValueError("Implausible future simulated Kelvin")
            xs.append(x[None]);ys.append(y[None]);years.append(yr);months.append(month)
        if len(xs)!=72:raise ValueError("Missing scheduled SSP source-target pairs")
        return np.stack(xs),np.stack(ys),np.array(years),np.array(months)
    finally:
        ds.close()

def score(pred,target,months,frozen):
    return {"weighted":latitude_weighted_metrics(pred,target,frozen["lat"]),
            "monthly_hot":monthly_local_hot(pred,target,months,frozen["p95"],frozen["lat"])}

def aggregate(rows):
    if len(rows)!=3:
        raise ValueError("Three frozen initialization seeds required")
    keys={"weighted_rmse_K":("weighted","rmse"),
          "weighted_mae_K":("weighted","mae"),
          "weighted_bias_K":("weighted","bias"),
          "local_p95_recall":("monthly_hot","hot_recall"),
          "local_p95_precision":("monthly_hot","hot_precision"),
          "predicted_hot_fraction":("monthly_hot","predicted_hot_fraction")}
    result={}
    for label,(group,metric) in keys.items():
        arr=[x[group][metric] for x in rows]
        if any(z is None or not np.isfinite(z) for z in arr):
            raise ValueError("Invalid CAMS SSP score: "+label)
        result[label]={"individual":list(map(float,arr)),
                      "mean":float(np.mean(arr)),
                      "sd_across_seeds":float(np.std(arr,ddof=1))}
    return result

def evaluate(frozen,manifest):
    import torch
    row=manifest_for_CAMS(manifest,frozen)
    out={"status":OUTPUT_STATUS,"source_model":"CAMS-CSM1-0",
         "source_member":frozen["member"],"grid":frozen["grid"],
         "native_calendar":frozen["calendar"],
         "regional_grid_shape":list(frozen["shape"]),
         "historical_source_sha256":frozen["historical_sha"],
         "checkpoint_sha":{f"{a}_{w}_{s}":p["sha256"]
                     for (a,w,s),p in frozen["checkpoints"].items()},
         "future_years":list(YEARS),"future_month_days":list(DAYS),
         "ssp_used_for_fitting_or_checkpoint_selection":False,
         "scenario_results":{},
         "limitations":["3×24 sparse target days per SSP, not full-year climate forecasts",
                        "GCM projection is simulated, not observed future weather",
                        "Historical 2005–2014 used for checkpoint selection, not independent testing",
                        "Only one CAMS ensemble member, not population-level climate inference",
                        "No future SSP checkpoints, normalizers or P95 re-fitted"]}
    torch.set_num_threads(1)
    for scenario in SSPS:
        x,y,years,months=read_native_future(row[scenario]["store"],scenario,frozen)
        groups={}
        for label,mask in [("all",np.ones(len(years),dtype=bool))]+[
            (str(year),years==year) for year in YEARS]:
            persistence=score(x[mask],y[mask],months[mask],frozen)
            models={}
            for model_name in MODELS:
                models[model_name]={}
                for weight in WEIGHTS:
                    rows=[]
                    for seed in SEEDS:
                        ck=frozen["checkpoints"][model_name,weight,seed]["data"]
                        model=(FNO2d(width=8,modes_y=4,modes_x=4,depth=2)
                            if model_name=="fno" else UNet2d(width=8))
                        model.load_state_dict(ck["state_dict"],strict=True)
                        model.eval()
                        normalized=(x[mask]-frozen["mu"])/frozen["sd"]
                        with torch.no_grad():
                            pred=np.concatenate([model(torch.from_numpy(batch)).numpy()
                                for batch in np.array_split(normalized,max(1,(len(normalized)+15)//16))])
                            pred=pred*frozen["sd"]+frozen["mu"]
                        rows.append(score(pred,y[mask],months[mask],frozen))
                    models[model_name][str(weight)]=aggregate(rows)
            groups[label]={"n_pairs":int(mask.sum()),
                           "persistence":persistence,"models":models}
        out["scenario_results"][scenario]={"store":row[scenario]["store"],
                                          "scores":groups}
    return out

def main():
    p=argparse.ArgumentParser()
    for key in ("historical","metadata","arm-dir","output"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    result_path=Path(a.output)
    if result_path.exists():raise FileExistsError("Never overwrite immutable SSP evidence")
    # Audit all Historical and checkpoint data before any Zarr future field is opened.
    frozen=inspect_frozen(a.historical,a.metadata,a.arm_dir)
    catalog,sha=read_catalog()
    manifest=select_pairs(catalog,sha,max_models=12)
    report=evaluate(frozen,manifest)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    result_path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],
        "scores":{s:{"pairs":r["scores"]["all"]["n_pairs"],
           "persistence_rmse_K":r["scores"]["all"]["persistence"]["weighted"]["rmse"],
           "fno1_rmse_K":r["scores"]["all"]["models"]["fno"]["0"]["weighted_rmse_K"]["mean"]}
           for s,r in report["scenario_results"].items()}},indent=2))
if __name__=="__main__":main()
