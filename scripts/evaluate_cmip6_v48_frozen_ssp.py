"""Strictly read-only SSP evaluation of ALL frozen v4.7 AWI Historical checkpoints.

SSP dates were preregistered in v4.4 (2030, 2050, 2080; days 8 and
20 of each month). No trained parameters, normalization, threshold or
checkpoint can change after reading SSP values. Sparse projected days,
not continuous climate trajectories or observed future forecast truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from climate_shift.data import load_npz_pairs
from climate_shift.metrics import latitude_weighted_metrics
from climate_shift.seasonal_extremes import fit_local_monthly_p95, local_monthly_hot_diagnostics
from climate_shift.models import FNO2d
from climate_shift.unet import UNet2d

try:
    from .audit_cmip6_awi_daily_pairs import contiguous_span, exact_daily_pair_indices
    from .run_cmip6_v47_historical_arms import REQUIRED_SHA, SEEDS, WEIGHTS, EPOCHS
except ImportError:
    from audit_cmip6_awi_daily_pairs import contiguous_span, exact_daily_pair_indices
    from run_cmip6_v47_historical_arms import REQUIRED_SHA, SEEDS, WEIGHTS, EPOCHS

YEARS = (2030, 2050, 2080)
DAYS = (8, 20)
SCENARIOS = ("ssp245", "ssp585")
MODEL_NAMES = ("fno", "unet")


def plan():
    return tuple((yr, month, day)
                 for yr in YEARS for month in range(1, 13) for day in DAYS)


def read_frozen_checkpoints(root: Path, historical_npz: Path):
    import torch
    if hashlib.sha256(historical_npz.read_bytes()).hexdigest() != REQUIRED_SHA:
        raise ValueError("Historical source SHA mismatch: refuse future test")
    x, y, years=load_npz_pairs(historical_npz)
    if x.shape != (1140,1,26,27) or years.min()!=1980 or years.max()!=2014:
        raise ValueError("Historical-only source contract violated")
    with np.load(historical_npz,allow_pickle=False) as z:
        lat=np.asarray(z["latitude"],dtype=float)
        times=z["target_times"]
    train=years<=2004
    mu=float(y[train].mean()); sigma=float(y[train].std())
    p95,reference=fit_local_monthly_p95(y[train],times[train],min_samples_per_month=20)
    if set(reference["actual_samples_per_month"].values())!={75}:
        raise ValueError("Training-only local P95 count violated")
    expected=[]
    all_checkpoints={}
    source_config=(round(mu,7),round(sigma,7))
    for weight in WEIGHTS:
        arm=root/f"weight_{int(weight)}"
        summ=json.loads((arm/"arm_summary.json").read_text())
        if (summ["status"]!="frozen_cmip6_historical_validation_only_no_ssp_evaluation"
            or summ["data_sha256"]!=REQUIRED_SHA or summ["seed_ids"]!=list(SEEDS)
            or summ["tail_weight"]!=weight or summ["epochs"]!=EPOCHS
            or summ["ssp_data_read"] is not False
            or set(summ["models"])!=set(MODEL_NAMES)):
            raise ValueError("Historical arm source/seed/weight provenance mismatch")
        for name in MODEL_NAMES:
            checkpoints=summ["models"][name]["checkpoints"]
            if {r["seed"] for r in checkpoints}!=set(SEEDS) or len(checkpoints)!=len(SEEDS):
                raise ValueError("Incomplete or duplicate model seeds")
            for rec in checkpoints:
                path=arm/rec["file"]
                if hashlib.sha256(path.read_bytes()).hexdigest()!=rec["sha256"]:
                    raise ValueError("Model checkpoint SHA mismatch")
                ckpt=torch.load(path,map_location="cpu",weights_only=True)
                if (ckpt["training_npz_sha256"]!=REQUIRED_SHA or
                    ckpt["architecture"]!=name or
                    ckpt["seed"]!=rec["seed"] or
                    ckpt["tail_weight"]!=weight or
                    ckpt["model_width"]!=8 or
                    ckpt["fno_modes"]!=4 or
                    ckpt["train_year_end"]!=2004 or
                    ckpt["validation_year_end"]!=2014 or
                    abs(ckpt["normalization_mean_K"]-mu)>1e-5 or
                    abs(ckpt["normalization_sd_K"]-sigma)>1e-5 or
                    ckpt["training_only_monthly_p95_K"].shape!=(12,26,27) or
                    not np.allclose(ckpt["training_only_monthly_p95_K"].numpy(),p95,atol=5e-5)):
                    raise ValueError("Frozen Historical checkpoint does not match fit-only reference")
                key=(name,int(weight),int(rec["seed"]))
                if key in all_checkpoints:
                    raise ValueError("Duplicate frozen checkpoint")
                all_checkpoints[key]=(ckpt,rec["sha256"])
                expected.append(key)
    if len(expected)!=18:
        raise ValueError("Must validate all 18 checkpoints before opening SSP data")
    return {"models":all_checkpoints,"train_mean":mu,"train_std":sigma,
            "local_p95":p95,"latitude":lat,"reference":reference,
            "checkpoint_sha256_by_arm":{
                f"{name}_weight{weight}_seed{seed}":sha
                for (name,weight,seed),(_,sha) in all_checkpoints.items()}}


def manifest_entry(manifest):
    if manifest.get("status")!="catalog_metadata_only_no_climate_fields_or_transfer_skill":
        raise ValueError("Require archived audited catalog")
    rows=[m for m in manifest.get("selected_candidates",[])
          if (m["source_id"],m["member_id"],m["grid_label"])==
             ("AWI-CM-1-1-MR","r1i1p1f1","gn")]
    if len(rows)!=1:
        raise ValueError("Require exactly one audited AWI model-member-grid")
    for scenario in SCENARIOS:
        url=rows[0][scenario]["store"]
        if (not url.startswith("gs://cmip6/") or
            f"/{scenario}/" not in url or "/day/tas/" not in url):
            raise ValueError("Invalid or unexpected future source")
    return rows[0]


def read_future_once(store, scenario, historical_latitude):
    import fsspec
    import xarray as xr
    if scenario not in SCENARIOS:
        raise ValueError("Invalid scenario")
    mapper=fsspec.get_mapper(store,anon=True,token="anon")
    ds=xr.open_zarr(mapper,consolidated=True,decode_times=False,
                    chunks=None,mask_and_scale=True)
    try:
        v=ds["tas"]
        if v.attrs.get("units")!="K" or set(v.dims)!={"time","lat","lon"}:
            raise ValueError("Expected daily mean Kelvin TAS native grid")
        lat=np.asarray(ds["lat"].values,dtype=float)
        lon=np.asarray(ds["lon"].values,dtype=float)%360
        i_lat=contiguous_span(lat,25,50)
        i_lon=contiguous_span(lon,235,260)
        native_lat=lat[i_lat]
        if (len(native_lat)!=26 or i_lon.stop-i_lon.start!=27
                or not np.allclose(native_lat,historical_latitude,atol=1e-8)):
            raise ValueError("Future native source latitude grid differs from Historical")
        time=ds["time"]
        units=str(time.attrs.get("units",""))
        calendar=str(time.attrs.get("calendar",time.encoding.get("calendar","standard")))
        if calendar!="proleptic_gregorian" or "since" not in units:
            raise ValueError("Unsupported changed native calendar or CF units")
        numeric=np.asarray(time.values,dtype=float)
        x,y,times,yearlist=[],[],[],[]
        for year,month,day in plan():
            i,j,source,target=exact_daily_pair_indices(numeric,(year,month,day),units,calendar)
            if source[:10]!=f"{year:04d}-{month:02d}-{day:02d}" or target[:4]!=str(year):
                raise ValueError("Source/target year or scheduled day mismatch")
            xx=np.asarray(v.isel(time=i,lat=i_lat,lon=i_lon).values,dtype=np.float32)
            yy=np.asarray(v.isel(time=j,lat=i_lat,lon=i_lon).values,dtype=np.float32)
            if xx.shape!=(26,27) or yy.shape!=(26,27) or not np.isfinite(xx).all() or not np.isfinite(yy).all():
                raise ValueError("Future sample shape, bounds or missingness error")
            if min(xx.min(),yy.min())<150 or max(xx.max(),yy.max())>350:
                raise ValueError("Future sample Kelvin out of physically possible range")
            x.append(xx[None]);y.append(yy[None])
            times.append(np.datetime64(target[:10]+"T12:00:00"))
            yearlist.append(year)
        assert len(x)==72
        return (np.stack(x),np.stack(y),
                np.asarray(yearlist,dtype=int),
                np.asarray(times,dtype="datetime64[ns]"))
    finally:
        ds.close()


def score(pred,y,times,lat,p95):
    out=latitude_weighted_metrics(pred,y,lat)
    tail=local_monthly_hot_diagnostics(pred,y,times,p95,latitude=lat)
    return {"weighted":out,"local_p95":tail}


def summarize_seed_scores(scores, baseline):
    if len(scores)!=3:
        raise ValueError("No incomplete seed comparison allowed")
    paths={
        "weighted_rmse_K":("weighted","rmse"),
        "weighted_mae_K":("weighted","mae"),
        "weighted_bias_K":("weighted","bias"),
        "local_p95_recall":("local_p95","hot_recall"),
        "local_p95_precision":("local_p95","hot_precision"),
        "predicted_hot_fraction":("local_p95","predicted_hot_fraction"),
    }
    output={}
    for label,(group,metric) in paths.items():
        values=[s[group][metric] for s in scores]
        if any(v is None or not np.isfinite(v) for v in values):
            raise ValueError(f"Missing or nonfinite SSP metric {label}")
        output[label]={"mean":float(np.mean(values)),
                       "individual":[float(x) for x in values],
                       "sd_across_seeds":float(np.std(values,ddof=1))}
    output["persistence_weighted_rmse_K"]=baseline["weighted"]["rmse"]
    return output


def evaluate(frozen, manifest):
    import torch
    model_row=manifest_entry(manifest)
    lat=frozen["latitude"]
    mu=frozen["train_mean"]; sigma=frozen["train_std"]
    p95=frozen["local_p95"]
    result={"status":"first_frozen_historical_to_sparse_SSP_forecast_evaluation",
            "historical_npz_sha256":REQUIRED_SHA,
            "historical_training_only_p95_samples_per_month":75,
            "historical_training_years":[1980,2004],
            "historical_validation_years":[2005,2014],
            "future_years":list(YEARS),
            "future_month_days":list(DAYS),
            "model":"AWI-CM-1-1-MR","member":"r1i1p1f1","grid":"gn",
            "source_catalog_sha256":manifest["catalog_sha256"],
            "checkpoint_sha256":frozen["checkpoint_sha256_by_arm"],
            "ssp_used_for_fitting_or_selection":False,
            "scenarios":{},
            "limits":[
                "This is one CMIP6 model-member simulation, not observed future conditions",
                "Only two fixed daily samples per month in three selected SSP years",
                "No continuous heatwave event duration or full-century SSP forecast scores",
                "History checkpoint selection used 2005–2014, so that is not an independent Historical test",
                "No hyperparameter optimization on future SSP scores is allowed",
                "Checkpoints of every loss candidate evaluated, not only historically preferred arm",
            ]}
    for scenario in SCENARIOS:
        x,y,years,times=read_future_once(model_row[scenario]["store"],scenario,lat)
        groups={}
        for scope,selection in (
            ("all",np.ones(len(years),dtype=bool)),
            *((str(yr),years==yr) for yr in YEARS),
        ):
            persistence=score(x[selection],y[selection],times[selection],lat,p95)
            model_results={}
            for name in MODEL_NAMES:
                model_results[name]={}
                for weight in WEIGHTS:
                    seed_metrics=[]
                    for seed in SEEDS:
                        ckpt,_=frozen["models"][(name,int(weight),int(seed))]
                        model=(FNO2d(width=8,modes_y=4,modes_x=4,depth=2) if name=="fno"
                               else UNet2d(width=8))
                        model.load_state_dict(ckpt["state_dict"],strict=True)
                        model.eval()
                        predictions=[]
                        with torch.no_grad():
                            normalized=(x[selection]-mu)/sigma
                            for chunk in np.array_split(normalized,max(1,(len(normalized)+15)//16)):
                                predicted=model(torch.from_numpy(chunk)).numpy()*sigma+mu
                                predictions.append(predicted)
                        pred=np.concatenate(predictions)
                        seed_metrics.append(score(pred,y[selection],times[selection],lat,p95))
                    model_results[name][str(int(weight))]=summarize_seed_scores(seed_metrics,persistence)
            groups[scope]={"n_pairs":int(selection.sum()),"persistence":persistence,
                           "models":model_results}
        result["scenarios"][scenario]={
            "store":model_row[scenario]["store"],
            "groups":groups,
        }
    if set(result["scenarios"])!=set(SCENARIOS):
        raise ValueError("Both scenarios mandatory")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--historical",required=True)
    p.add_argument("--arm-dir",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    output=Path(a.output)
    if output.exists():
        raise FileExistsError("Do not overwrite SSP independent evaluation")
    frozen=read_frozen_checkpoints(Path(a.arm_dir),Path(a.historical))
    manifest=json.loads(Path(a.manifest).read_text())
    report=evaluate(frozen,manifest)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":report["status"],"future_years":report["future_years"],
        "scenarios":{
            k:{"samples":v["groups"]["all"]["n_pairs"],
               "persistence_RMSE_K":v["groups"]["all"]["persistence"]["weighted"]["rmse"],
               "fno_by_tail_weight":{
                   w:round(v["groups"]["all"]["models"]["fno"][w]["weighted_rmse_K"]["mean"],6)
                   for w in ("0","1","3")}}
            for k,v in report["scenarios"].items()}},indent=2))


if __name__=="__main__":
    main()
