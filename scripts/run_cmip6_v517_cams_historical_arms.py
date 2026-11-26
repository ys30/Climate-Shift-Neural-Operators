"""E1 v5.17: CAMS Historical-only FNO and U-Net training, frozen design.

Fits one predeclared tail-weight arm for all three seeds. Neither source
selection nor validation loads any future SSP array. Historical validation
2005–2014 is used for checkpoint selection, NOT untouched test data.
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
from scripts.export_cmip6_v516_cams_historical import verify_npz, STATUS as INPUT_STATUS
from scripts.export_cmip6_bcc_v51_historical import monthly_training_reference
from scripts.fit_cmip6_bcc_v52_historical import monthly_local_hot

SEEDS=(11,22,33)
WEIGHTS=(0,1,3)
EPOCHS=100
STATUS="v517_CAMS_independent_Historical_only_frozen_three_seeds_and_models"

def validate_source(npz,metadata):
    data=Path(npz);meta=json.loads(Path(metadata).read_text())
    digest=hashlib.sha256(data.read_bytes()).hexdigest()
    if meta.get("status")!=INPUT_STATUS or meta.get("npz_sha256")!=digest:
        raise ValueError("Immutable v5.16 CAMS Historical artifact checksum mismatch")
    if (meta.get("source_id")!="CAMS-CSM1-0" or meta.get("ssp_data_opened") is not False
        or meta.get("model_trained") is not False
        or meta.get("train_pairs")!=900 or meta.get("validation_pairs")!=240
        or meta.get("train_only_monthly_p95") is not True):
        raise ValueError("Untrusted CAMS source provenance")
    shape=tuple(meta["native_grid_shape"])
    verify_npz(data,shape)
    with np.load(data,allow_pickle=False) as d:
        x=d["X"].astype(np.float32)
        y=d["y"].astype(np.float32)
        months=d["months"].astype(np.int32)
        years=d["years"].astype(np.int32)
        lat=d["latitude"].astype(float)
        train=d["train_mask"].astype(bool)
        p95=d["monthly_train_p95_K"].astype(np.float32)
        src=d["source_cf_day_value"].astype(float)
        tgt=d["target_cf_day_value"].astype(float)
    expected,counts=monthly_training_reference(y,years,months,train)
    if not np.allclose(expected,p95,atol=5e-5,rtol=0):
        raise ValueError("CAMS monthly P95 must come from historical training targets only")
    if (x.shape!=(1140,1,*shape) or train.sum()!=900 or (~train).sum()!=240
        or not tgt[train].max()<src[~train].min()):
        raise ValueError("Incorrect historical boundaries")
    mu=float(y[train].mean());sd=float(y[train].std())
    if not np.isfinite(mu) or not np.isfinite(sd) or sd<1e-6:
        raise ValueError("Invalid independent Historical-only CAMS normalizer")
    return dict(x=x,y=y,lat=lat,months=months,train=train,p95=p95,mu=mu,sd=sd,
                digest=digest,shape=shape,calendar=meta["native_calendar"],counts=counts)

def train_arm(source,root,weight,epochs=EPOCHS):
    import torch
    if weight not in WEIGHTS or epochs!=EPOCHS:
        raise ValueError("Only preregistered 100-epoch Historical weight arms")
    root=Path(root)
    if root.exists():raise FileExistsError("Cannot overwrite frozen CAMS model arm")
    d=source
    x,y,train=d["x"],d["y"],d["train"]
    mu,sd=d["mu"],d["sd"]
    xx=torch.from_numpy((x-mu)/sd)
    yy=torch.from_numpy((y-mu)/sd)
    p95_std=torch.from_numpy((d["p95"]-mu)/sd)
    mon=torch.from_numpy((d["months"]-1).astype(np.int64))
    tr=torch.from_numpy(np.flatnonzero(train))
    va=torch.from_numpy(np.flatnonzero(~train))
    torch.set_num_threads(1)
    root.mkdir(parents=True)
    summary={"status":STATUS,"gcm":"CAMS-CSM1-0","historical_sha256":d["digest"],
        "native_calendar":d["calendar"],"native_shape":list(d["shape"]),
        "seeds":list(SEEDS),"tail_weight":weight,"loss_multiplier":weight+1,
        "epochs":EPOCHS,"train_pairs":900,"validation_pairs":240,
        "normalizer_train_mean_K":mu,"normalizer_train_std_K":sd,
        "source_only_train_month_P95_counts":d["counts"],
        "ssp_fields_read":False,"future_checkpoint_selection":False,
        "persistence_validation":latitude_weighted_metrics(x[~train],y[~train],d["lat"]),
        "models":{}}
    for model_name in ("fno","unet"):
        individual=[]
        for seed in SEEDS:
            torch.manual_seed(seed)
            model=FNO2d(width=8,modes_y=4,modes_x=4,depth=2) if model_name=="fno" else UNet2d(width=8)
            opt=torch.optim.Adam(model.parameters(),lr=.001)
            best=float("inf");best_state=None;best_epoch=None
            for epoch in range(1,EPOCHS+1):
                model.train()
                for batch in tr[torch.randperm(len(tr))].split(8):
                    opt.zero_grad()
                    pred=model(xx[batch])
                    squared=(pred-yy[batch]).square()
                    if weight:
                        hot=yy[batch,0]>p95_std[mon[batch]]
                        loss=(squared*(1+weight*hot[:,None].to(squared.dtype))).mean()
                    else:loss=squared.mean()
                    if not torch.isfinite(loss):raise ValueError("Nonfinite CAMS training loss")
                    loss.backward();opt.step()
                model.eval()
                with torch.no_grad():
                    total=sum(float((model(xx[b])-yy[b]).square().sum()) for b in va.split(32))
                    mse=total/len(va)/np.prod(y.shape[1:])
                if not np.isfinite(mse):raise ValueError("Nonfinite CAMS validation MSE")
                if mse<best:
                    best=mse;best_epoch=epoch
                    best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best_state);model.eval()
            fname=f"{model_name}_cams_seed{seed}_weight{weight}.pt"
            path=root/fname
            torch.save({"state_dict":best_state,"model":model_name,
                "seed":seed,"tail_weight":weight,"epochs":EPOCHS,
                "best_epoch":best_epoch,"historical_npz_sha256":d["digest"],
                "normalization_mean_K":mu,"normalization_sd_K":sd,
                "native_grid_shape":list(d["shape"]),"calendar":d["calendar"],
                "training_only_monthly_p95_K":torch.from_numpy(d["p95"].copy()),
                "width":8,"fno_modes":4},path)
            with torch.no_grad():
                prediction=np.concatenate([model(xx[b]).numpy()
                       for b in va.split(32)],axis=0)*sd+mu
            scores=latitude_weighted_metrics(prediction,y[~train],d["lat"])
            tail=monthly_local_hot(prediction,y[~train],d["months"][~train],
                                    d["p95"],d["lat"])
            individual.append({"seed":seed,"checkpoint":fname,
                "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "best_epoch":best_epoch,"parameter_count":sum(p.numel() for p in model.parameters()),
                "weighted":scores,"tail":tail})
        if len(individual)!=3 or len({r["sha256"] for r in individual})!=3:
            raise ValueError("Missing or duplicate frozen CAMS checkpoints")
        def aggregate(section,field):
            v=[r[section][field] for r in individual]
            if any(x is None or not np.isfinite(x) for x in v):
                raise ValueError("Nonfinite CAMS validation metric")
            return {"individual":list(map(float,v)),"mean":float(np.mean(v)),
                    "sd_seeds":float(np.std(v,ddof=1))}
        summary["models"][model_name]={
            "checkpoints":individual,
            "weighted_rmse_K":aggregate("weighted","rmse"),
            "weighted_mae_K":aggregate("weighted","mae"),
            "weighted_bias_K":aggregate("weighted","bias"),
            "local_p95_recall":aggregate("tail","hot_recall"),
            "local_p95_precision":aggregate("tail","hot_precision")}
    (root/"cams_arm_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser()
    for name in ("input","metadata","output-dir"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--tail-weight",type=int,choices=WEIGHTS,required=True)
    a=p.parse_args()
    src=validate_source(a.input,a.metadata)
    report=train_arm(src,a.output_dir,a.tail_weight)
    print(json.dumps({"status":report["status"],"tail_weight":report["tail_weight"],
        "models":{m:{"rmse":x["weighted_rmse_K"]["mean"],
                      "recall":x["local_p95_recall"]["mean"]}
                  for m,x in report["models"].items()}},indent=2))
if __name__=="__main__":main()
