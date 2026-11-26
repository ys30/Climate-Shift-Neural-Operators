"""Full frozen BCC Historical native-noleap FNO/U-Net training arms.

Exactly three 100-epoch seeds per predeclared loss multiplier 1x, 2x, 4x.
No future SSP arrays, thresholds or tuning. Use BCC-only v5.1 SHA.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .fit_cmip6_bcc_v52_historical import fit, validate_historical
except ImportError:
    from fit_cmip6_bcc_v52_historical import fit, validate_historical

SEEDS=(11,22,33)
TAIL_WEIGHTS=(0.0,1.0,3.0)
EPOCHS=100
BCC_V51_SHA="a23260e8f14a23fb57483038bcd6ca03ba74c7944a48ba29d62bcc5bd354f452"


def aggregate(reports,weight):
    if weight not in TAIL_WEIGHTS or len(reports)!=3 or [r["seed"] for r in reports]!=list(SEEDS):
        raise ValueError("Require all frozen BCC training seeds and weight arm")
    if any(r["source_npz_sha256"]!=BCC_V51_SHA or r["epochs"]!=EPOCHS or
           r["tail_weight"]!=weight or r["calendar"]!="noleap" or
           r["grid_shape"]!=[23,23] or r["ssp_data_read"] is not False for r in reports):
        raise ValueError("BCC fit provenance or SSP isolation violated")
    baselines=[r["persistence_validation"] for r in reports]
    if any(b!=baselines[0] for b in baselines):
        raise ValueError("Historical Persistence changed across seeds")
    output={}
    for name in ("fno","unet"):
        output[name]={}
        paths={
            "weighted_rmse_K":("validation_weighted","rmse"),
            "weighted_mae_K":("validation_weighted","mae"),
            "weighted_bias_K":("validation_weighted","bias"),
            "local_p95_recall":("validation_monthly_hot","hot_recall"),
            "local_p95_precision":("validation_monthly_hot","hot_precision"),
            "predicted_hot_fraction":("validation_monthly_hot","predicted_hot_fraction"),
        }
        for label,(grp,key) in paths.items():
            vals=[r["models"][name][grp][key] for r in reports]
            if any(v is None or not np.isfinite(v) for v in vals):
                raise ValueError(f"Invalid BCC validation metric: {label}")
            output[name][label]={
                "individual":[float(v) for v in vals],
                "mean":float(np.mean(vals)),
                "sd_across_initialization_seeds":float(np.std(vals,ddof=1)),
            }
        count=[r["models"][name]["parameter_count"] for r in reports]
        if len(set(count))!=1:raise ValueError("Model architecture changed among seeds")
        output[name]["parameter_count"]=count[0]
        output[name]["best_epoch_by_seed"]=[r["models"][name]["best_epoch"] for r in reports]
        output[name]["checkpoints"]=[
            {"seed":r["seed"],
             "file":f"seed_{r['seed']}/{r['models'][name]['checkpoint']}",
             "sha256":r["models"][name]["checkpoint_sha256"]}
            for r in reports]
    return {
        "status":"BCC_historical_validation_only_full_frozen_arms_no_future_skill",
        "model":"BCC-CSM2-MR","member":"r1i1p1f1","grid":"gn",
        "native_calendar":"noleap","native_grid_shape":[23,23],
        "source_npz_sha256":BCC_V51_SHA,
        "train_years":[1980,2004],"validation_years":[2005,2014],
        "train_count":900,"validation_count":240,
        "seeds":list(SEEDS),"epochs":EPOCHS,"tail_weight":weight,
        "hot_training_multiplier":1+weight,
        "validation_persistence":baselines[0],
        "models":output,"ssp_data_read":False,
        "checkpoint_selection":"unweighted Historical-only 2005–2014 validation MSE",
        "limitations":[
            "Only three dates/month/year in Historical training; no continuous heatwaves",
            "Validation is used for checkpoint selection, not untouched independent test",
            "BCC native noleap and 23x23 requires its own calibration and forecasting",
            "No BCC SSP climate-change forecast accuracy has yet been measured",
        ],
    }


def run(npz,output,weight):
    if weight not in TAIL_WEIGHTS:
        raise ValueError("Unfrozen BCC loss multiplier")
    path=Path(npz)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=BCC_V51_SHA:raise ValueError("BCC source hash changed")
    validate_historical(path,sha)
    root=Path(output)
    if root.exists():raise FileExistsError("Existing model arm must never be overwritten")
    results=[]
    for seed in SEEDS:
        r=fit(str(path),sha,str(root/f"seed_{seed}"),seed=seed,epochs=EPOCHS,
              tail_weight=weight)
        results.append(r)
    report=aggregate(results,weight)
    (root/"bcc_arm_summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--tail-weight",type=float,required=True,choices=TAIL_WEIGHTS)
    a=p.parse_args()
    r=run(a.input,a.output_dir,a.tail_weight)
    print(json.dumps({"status":r["status"],"weight":r["tail_weight"],"seeds":r["seeds"],
       "validation_persistence_rmse_K":r["validation_persistence"]["rmse"],
       "models":{name:{"rmse":m["weighted_rmse_K"]["mean"],
                        "recall":m["local_p95_recall"]["mean"],
                        "precision":m["local_p95_precision"]["mean"]}
                 for name,m in r["models"].items()}},indent=2))


if __name__=="__main__":
    main()
