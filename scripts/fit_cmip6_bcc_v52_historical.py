"""BCC Historical-only noleap-aware FNO/U-Net smoke fit.

Uses frozen native-calendar BCC artifact, and never loads ScenarioMIP data.
Model sizes and losses match the AWI FNO/U-Net experiment, but the BCC grid,
normalization and training-only monthly P95 are entirely independent.
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
    from .export_cmip6_bcc_v51_historical import verify_saved_npz, monthly_training_reference
except ImportError:
    from export_cmip6_bcc_v51_historical import verify_saved_npz, monthly_training_reference


def validate_historical(path: Path, sha256: str):
    if len(sha256)!=64 or sha256!=hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("BCC immutable historical artifact hash mismatch")
    verify_saved_npz(path)
    with np.load(path,allow_pickle=False) as d:
        x=d["X"].astype(np.float32)
        y=d["y"].astype(np.float32)
        yrs=d["years"].astype(int)
        months=d["months"].astype(int)
        lat=d["latitude"].astype(float)
        origin=d["source_cf_day_value"].astype(float)
        target=d["target_cf_day_value"].astype(float)
        train=d["train_mask"].astype(bool)
        p95_saved=d["monthly_train_p95_K"]
    if (train.sum(),(~train).sum())!=(900,240) or not np.all(target-origin==1):
        raise ValueError("No leap/calendar or source-target data isolation violation")
    if not target[train].max()<origin[~train].min():
        raise ValueError("BCC native forecast train/validation contamination")
    p95,counts=monthly_training_reference(y,yrs,months,train)
    if not np.allclose(p95,p95_saved,rtol=0,atol=5e-5):
        raise ValueError("BCC P95 not reproduced from Historical targets only")
    return x,y,yrs,months,lat,train,p95,counts


def monthly_local_hot(pred,target,months,p95,lat):
    """Area weighted BCC exceedance, months are native noleap identifiers."""
    if pred.shape!=target.shape or pred.ndim!=4 or pred.shape[1]!=1:
        raise ValueError("Expected matched BCC prediction fields")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite evaluation grid")
    q=p95[np.asarray(months,dtype=int)-1,None]
    w=np.broadcast_to(np.cos(np.deg2rad(lat))[None,None,:,None],target.shape)
    hot=target>q;forecast=pred>q
    ntrue=float(np.sum(w*hot));npred=float(np.sum(w*forecast))
    hits=float(np.sum(w*(hot & forecast)))
    return {
        "reference_hot_fraction":ntrue/float(np.sum(w)),
        "predicted_hot_fraction":npred/float(np.sum(w)),
        "hot_recall":hits/ntrue if ntrue>0 else None,
        "hot_precision":hits/npred if npred>0 else None,
        "threshold_definition":"BCC training-only native-month local P95",
        "weighting":"cos(latitude)",
    }


def fit(npz: str, expected_sha: str, output: str, seed=11, epochs=2,
        tail_weight=0.0, width=8, modes=4):
    import torch
    if (seed not in (11,22,33) or not 1<=epochs<=100 or
        tail_weight not in (0.,1.,3.) or width!=8 or modes!=4):
        raise ValueError("Unfrozen BCC training setting")
    out=Path(output)
    if out.exists():
        raise FileExistsError("Never overwrite BCC checkpoint directory")
    x,y,years,months,lat,train,p95,counts=validate_historical(Path(npz),expected_sha)
    mu=float(y[train].mean());sigma=float(y[train].std())
    if sigma<1e-6:raise ValueError("BCC training data near constant")
    xx=torch.from_numpy((x-mu)/sigma)
    yy=torch.from_numpy((y-mu)/sigma)
    p95_std=torch.from_numpy(((p95-mu)/sigma).astype(np.float32))
    mi=torch.from_numpy((months-1).astype(np.int64))
    tr=torch.from_numpy(np.flatnonzero(train))
    va=torch.from_numpy(np.flatnonzero(~train))
    torch.set_num_threads(1)
    out.mkdir(parents=True)
    summary={
        "status":"bcc_noleap_Historical_only_trained_no_future_SSP",
        "source_npz_sha256":expected_sha,
        "calendar":"noleap","grid_shape":[23,23],
        "train_years":[1980,2004],"validation_years":[2005,2014],
        "training_count":int(train.sum()),"validation_count":int((~train).sum()),
        "p95_train_month_counts":counts,
        "seed":seed,"epochs":epochs,"tail_weight":tail_weight,
        "normalization_mean_K":mu,"normalization_sd_K":sigma,
        "ssp_data_read":False,
        "persistence_validation":latitude_weighted_metrics(x[~train],y[~train],lat),
        "models":{},
    }
    for name in ("fno","unet"):
        torch.manual_seed(seed)
        model=(FNO2d(width=width,modes_y=modes,modes_x=modes,depth=2)
               if name=="fno" else UNet2d(width=width))
        optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
        best_loss=float("inf");best_epoch=None;best_state=None;history=[]
        for epoch in range(1,epochs+1):
            model.train();running=0.
            order=tr[torch.randperm(len(tr))]
            for batch in order.split(8):
                optimizer.zero_grad()
                pred=model(xx[batch])
                squared=(pred-yy[batch]).square()
                if tail_weight:
                    event=yy[batch,0]>p95_std[mi[batch]]
                    loss=(squared*(1+tail_weight*event[:,None].to(squared.dtype))).mean()
                else:
                    loss=squared.mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite BCC loss")
                loss.backward();optimizer.step()
                running+=float(loss.detach())*len(batch)
            model.eval()
            with torch.no_grad():
                total=0.
                for batch in va.split(32):
                    total+=float((model(xx[batch])-yy[batch]).square().sum())
                val_mse=total/len(va)/np.prod(y.shape[1:])
            if not np.isfinite(val_mse):
                raise ValueError("Nonfinite validation MSE")
            history.append({"epoch":epoch,"train_loss":running/len(tr),
                            "validation_mse_standardized":val_mse})
            if val_mse<best_loss:
                best_loss=val_mse;best_epoch=epoch
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state)
        ckpt=out/f"{name}_bcc_seed{seed}_weight{int(tail_weight)}.pt"
        torch.save({
            "state_dict":best_state, "model":name,
            "seed":seed,"tail_weight":tail_weight,"best_epoch":best_epoch,
            "historical_npz_sha256":expected_sha,
            "calendar":"noleap","grid_shape":[23,23],
            "normalization_mean_K":mu,"normalization_sd_K":sigma,
            "training_only_monthly_p95_K":torch.from_numpy(p95.astype(np.float32)),
            "width":width,"fno_modes":modes,
        },ckpt)
        saved=torch.load(ckpt,map_location="cpu",weights_only=True)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        with torch.no_grad():
            pp=np.concatenate([model(xx[b]).numpy() for b in va.split(32)],axis=0)*sigma+mu
        summary["models"][name]={
            "checkpoint":ckpt.name,
            "checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            "best_epoch":best_epoch,
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "history":history,
            "validation_weighted":latitude_weighted_metrics(pp,y[~train],lat),
            "validation_monthly_hot":monthly_local_hot(pp,y[~train],months[~train],p95,lat),
        }
    (out/"bcc_historical_smoke.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--seed",type=int,default=11)
    p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--tail-weight",type=float,default=0.)
    args=p.parse_args()
    result=fit(args.input,args.expected_sha,args.output_dir,args.seed,args.epochs,args.tail_weight)
    print(json.dumps({
        "status":result["status"],
        "validation_persistence_rmse_K":result["persistence_validation"]["rmse"],
        "models":{k:{"rmse_K":v["validation_weighted"]["rmse"],
                     "p95_recall":v["validation_monthly_hot"]["hot_recall"],
                     "best_epoch":v["best_epoch"]}
                  for k,v in result["models"].items()},
    },indent=2))


if __name__=="__main__":
    main()
