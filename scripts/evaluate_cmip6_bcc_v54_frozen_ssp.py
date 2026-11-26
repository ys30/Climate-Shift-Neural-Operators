"""Blind BCC native-noleap Historical-checkpoint -> SSP245/585 score harness.

All 18 model checkpoints must be source/SHA verified *before* opening any
SSP Zarr. No weights, P95 thresholds or standardization fitted on future
simulations. Uses previously fixed 2030/2050/2080 day 8/20 monthly schedule.
These are sparse simulated SSP climate targets, not future observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from climate_shift.metrics import latitude_weighted_metrics
from climate_shift.models import FNO2d
from climate_shift.unet import UNet2d

try:
    from .audit_cmip6_awi_daily_pairs import contiguous_span,exact_daily_pair_indices
    from .audit_cmip6_bcc_regional_pairs import get_model
    from .export_cmip6_bcc_v51_historical import verify_saved_npz,monthly_training_reference
    from .fit_cmip6_bcc_v52_historical import monthly_local_hot
    from .run_cmip6_bcc_v53_historical_arms import SEEDS,TAIL_WEIGHTS,EPOCHS,BCC_V51_SHA
except ImportError:
    from audit_cmip6_awi_daily_pairs import contiguous_span,exact_daily_pair_indices
    from audit_cmip6_bcc_regional_pairs import get_model
    from export_cmip6_bcc_v51_historical import verify_saved_npz,monthly_training_reference
    from fit_cmip6_bcc_v52_historical import monthly_local_hot
    from run_cmip6_bcc_v53_historical_arms import SEEDS,TAIL_WEIGHTS,EPOCHS,BCC_V51_SHA

YEARS=(2030,2050,2080)
DAYS=(8,20)
SCENARIOS=("ssp245","ssp585")
MODEL_NAMES=("fno","unet")


def planned_days():
    return [(year,month,day) for year in YEARS
            for month in range(1,13) for day in DAYS]


def inspect_frozen_sources(historical_npz,arm_root):
    import torch
    historical_npz=Path(historical_npz)
    arm_root=Path(arm_root)
    if hashlib.sha256(historical_npz.read_bytes()).hexdigest()!=BCC_V51_SHA:
        raise ValueError("BCC Historical immutable NPZ SHA mismatch")
    verify_saved_npz(historical_npz)
    with np.load(historical_npz,allow_pickle=False) as d:
        y=d["y"].astype(np.float32)
        years=d["years"].astype(int)
        months=d["months"].astype(int)
        train=d["train_mask"].astype(bool)
        lat=d["latitude"].astype(float)
        lon=d["longitude_east"].astype(float)
        p95_saved=d["monthly_train_p95_K"].astype(np.float32)
    if not (train.sum()==900 and (~train).sum()==240 and
            years[train].min()==1980 and years[train].max()==2004 and
            years[~train].min()==2005 and years[~train].max()==2014):
        raise ValueError("Historical BCC train/validation source cohort contaminated")
    mu=float(y[train].mean());sd=float(y[train].std())
    p95,counts=monthly_training_reference(y,years,months,train)
    if set(counts.values())!={75} or not np.allclose(p95,p95_saved,atol=1e-5,rtol=0):
        raise ValueError("Training-only BCC native-month P95 mismatch")
    checkpoints={}
    for weight in TAIL_WEIGHTS:
        arm=arm_root/f"weight_{int(weight)}"
        summ=json.loads((arm/"bcc_arm_summary.json").read_text())
        if (summ["status"]!="BCC_historical_validation_only_full_frozen_arms_no_future_skill"
                or summ["source_npz_sha256"]!=BCC_V51_SHA
                or summ["native_calendar"]!="noleap"
                or summ["native_grid_shape"]!=[23,23]
                or summ["epochs"]!=EPOCHS or summ["tail_weight"]!=weight
                or summ["seeds"]!=list(SEEDS) or summ["ssp_data_read"] is not False):
            raise ValueError("Unfrozen BCC Historical training arm identity")
        for name in MODEL_NAMES:
            entries=summ["models"][name]["checkpoints"]
            if len(entries)!=3 or {r["seed"] for r in entries}!=set(SEEDS):
                raise ValueError("Incomplete historical BCC arm seeds")
            for rec in entries:
                path=arm/rec["file"]
                if hashlib.sha256(path.read_bytes()).hexdigest()!=rec["sha256"]:
                    raise ValueError("Changed BCC checkpoint file")
                ck=torch.load(path,map_location="cpu",weights_only=True)
                if (ck["model"]!=name or ck["seed"]!=rec["seed"] or
                    ck["tail_weight"]!=weight or
                    ck["historical_npz_sha256"]!=BCC_V51_SHA or
                    ck["calendar"]!="noleap" or ck["grid_shape"]!=[23,23] or
                    ck["width"]!=8 or ck["fno_modes"]!=4 or
                    abs(ck["normalization_mean_K"]-mu)>1e-5 or
                    abs(ck["normalization_sd_K"]-sd)>1e-5 or
                    not np.allclose(ck["training_only_monthly_p95_K"].numpy(),p95,atol=5e-5)):
                    raise ValueError("Checkpoint changed from BCC Historical-only fitting")
                key=(name,int(weight),int(rec["seed"]))
                if key in checkpoints:raise ValueError("Duplicate BCC checkpoint")
                checkpoints[key]={"checkpoint":ck,"sha":rec["sha256"]}
    if len(checkpoints)!=18:
        raise ValueError("All 18 BCC model checkpoints required before future read")
    return {
        "checkpoints":checkpoints,
        "mean":mu,"sd":sd,"p95":p95,"lat":lat,"lon":lon,
        "checkpoint_sha_by_model":{
            f"{name}_weight{weight}_seed{seed}":item["sha"]
            for (name,weight,seed),item in checkpoints.items()
        },
    }


def get_future_pairs(store,scenario,lat_ref,lon_ref):
    import fsspec
    import xarray as xr
    if scenario not in SCENARIOS or f"/{scenario}/" not in store:
        raise ValueError("Future source experiment mismatch")
    ds=xr.open_zarr(fsspec.get_mapper(store,anon=True,token="anon"),
                    consolidated=True,decode_times=False,chunks=None,
                    mask_and_scale=True)
    try:
        tas=ds["tas"]
        if tas.attrs.get("units")!="K" or set(tas.dims)!={"time","lat","lon"}:
            raise ValueError("Expected native BCC Kelvin daily mean tas")
        lat=np.asarray(ds["lat"].values,dtype=float)
        lon=np.asarray(ds["lon"].values,dtype=float)%360
        lat_sel=contiguous_span(lat,25.,50.)
        lon_sel=contiguous_span(lon,235.,260.)
        if not (np.array_equal(lat[lat_sel],lat_ref) and
                np.array_equal(lon[lon_sel],lon_ref)):
            raise ValueError("Future BCC grid coordinate values differ from Historical")
        t=ds["time"]
        calendar=str(t.attrs.get("calendar",t.encoding.get("calendar","standard")))
        units=str(t.attrs.get("units",""))
        if calendar not in ("noleap","365_day") or "days since" not in units:
            raise ValueError("Future BCC native noleap calendar mismatch")
        tnum=np.asarray(t.values,dtype=float)
        x,y,years,months=[],[],[],[]
        for yr,month,day in planned_days():
            i,j,src,dst=exact_daily_pair_indices(tnum,(yr,month,day),units,calendar)
            if src[:10]!=f"{yr:04d}-{month:02d}-{day:02d}" or dst[:4]!=str(yr):
                raise ValueError("SSP date/target mismatch")
            xx=np.asarray(tas.isel(time=i,lat=lat_sel,lon=lon_sel).values,dtype=np.float32)
            yy=np.asarray(tas.isel(time=j,lat=lat_sel,lon=lon_sel).values,dtype=np.float32)
            if xx.shape!=(23,23) or yy.shape!=(23,23) or not np.isfinite(xx).all() or not np.isfinite(yy).all():
                raise ValueError("Future grid has missing/out-of-grid values")
            if min(xx.min(),yy.min())<150 or max(xx.max(),yy.max())>350:
                raise ValueError("Future BCC physically invalid Kelvin")
            x.append(xx[None]);y.append(yy[None])
            years.append(yr);months.append(month)
        if len(x)!=72:raise ValueError("Missing frozen source calendar schedule")
        return np.stack(x),np.stack(y),np.array(years),np.array(months)
    finally:
        ds.close()


def one_score(pred,target,months,lat,p95):
    return {
        "weighted":latitude_weighted_metrics(pred,target,lat),
        "monthly_hot":monthly_local_hot(pred,target,months,p95,lat),
    }


def aggregate_seed_scores(rows):
    if len(rows)!=3:raise ValueError("All three frozen training seeds required")
    entries={
        "weighted_rmse_K":("weighted","rmse"),
        "weighted_mae_K":("weighted","mae"),
        "weighted_bias_K":("weighted","bias"),
        "local_p95_recall":("monthly_hot","hot_recall"),
        "local_p95_precision":("monthly_hot","hot_precision"),
        "predicted_hot_fraction":("monthly_hot","predicted_hot_fraction"),
    }
    result={}
    for name,(field,key) in entries.items():
        values=[r[field][key] for r in rows]
        if any(v is None or not np.isfinite(v) for v in values):
            raise ValueError(f"Invalid BCC future evaluation measure {name}")
        result[name]={"individual":[float(v) for v in values],
                      "mean":float(np.mean(values)),
                      "sd_across_seeds":float(np.std(values,ddof=1))}
    return result


def evaluate(frozen,manifest):
    import torch
    record=get_model(manifest)
    lat=frozen["lat"];lon=frozen["lon"];p95=frozen["p95"]
    mu=frozen["mean"];sd=frozen["sd"]
    report={
        "status":"first_independent_GCM_BCC_Historical_to_sparse_SSP_frozen_forecast",
        "historical_source_sha256":BCC_V51_SHA,
        "historical_training_years":[1980,2004],
        "historical_validation_years":[2005,2014],
        "future_years":list(YEARS),"future_month_days":list(DAYS),
        "source_model":"BCC-CSM2-MR","source_member":"r1i1p1f1",
        "grid":"gn","native_calendar":"noleap",
        "regional_grid_shape":[23,23],
        "checkpoint_sha":frozen["checkpoint_sha_by_model"],
        "ssp_used_for_fitting_or_checkpoint_selection":False,
        "scenario_results":{},
        "limits":[
            "Only one BCC source member, complementing but not completing AWI replication",
            "Two fixed dates/month in three selected years, not continuous SSP daily skill",
            "CMIP6 simulated projections are not observed future truth",
            "No SSP target gradients, threshold fitting or checkpoint selection",
            "Any cross-model effects require matched model-specific thresholds and uncertainty analysis",
        ],
    }
    for scenario in SCENARIOS:
        x,y,years,months=get_future_pairs(record[scenario]["store"],scenario,lat,lon)
        scores={}
        for scope,mask in [("all",np.ones(len(years),dtype=bool))]+[
            (str(year),years==year) for year in YEARS]:
            pers=one_score(x[mask],y[mask],months[mask],lat,p95)
            models={}
            for name in MODEL_NAMES:
                models[name]={}
                for weight in TAIL_WEIGHTS:
                    seed_results=[]
                    for seed in SEEDS:
                        ck=frozen["checkpoints"][(name,int(weight),int(seed))]["checkpoint"]
                        model=(FNO2d(width=8,modes_y=4,modes_x=4,depth=2)
                               if name=="fno" else UNet2d(width=8))
                        model.load_state_dict(ck["state_dict"],strict=True)
                        model.eval()
                        with torch.no_grad():
                            normalized=(x[mask]-mu)/sd
                            pred=np.concatenate([
                                model(torch.from_numpy(batch)).numpy()*sd+mu
                                for batch in np.array_split(normalized,
                                    max(1,(len(normalized)+15)//16))
                            ])
                        seed_results.append(one_score(pred,y[mask],months[mask],lat,p95))
                    models[name][str(int(weight))]=aggregate_seed_scores(seed_results)
            scores[scope]={
                "n_pairs":int(mask.sum()),
                "persistence":pers,
                "models":models,
            }
        report["scenario_results"][scenario]={
            "store":record[scenario]["store"],
            "scores":scores,
        }
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--historical",required=True)
    p.add_argument("--arm-dir",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError("Never overwrite SSP evaluation")
    # Strictly verify all frozen source/checkpoints before reading any SSP data.
    frozen=inspect_frozen_sources(a.historical,a.arm_dir)
    report=evaluate(frozen,json.loads(Path(a.manifest).read_text()))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":report["status"],
        "scenarios":{
            name:{"n_pairs":r["scores"]["all"]["n_pairs"],
                  "persistence_rmse_K":r["scores"]["all"]["persistence"]["weighted"]["rmse"],
                  "fno_rmse_K_by_weight":{
                      w:r["scores"]["all"]["models"]["fno"][w]["weighted_rmse_K"]["mean"]
                      for w in ("0","1","3")}}
            for name,r in report["scenario_results"].items()}
    },indent=2))


if __name__=="__main__":
    main()
