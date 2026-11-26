"""Fit FNO/U-Net ONLY on immutable AWI Historical daily TAS train/validation.

Requires validated v4.5 artifact; never opens any SSP inputs. This is a
bounded smoke train, not a projected future-climate test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from climate_shift.data import load_npz_pairs
from climate_shift.seasonal_extremes import fit_local_monthly_p95, local_monthly_hot_diagnostics
from climate_shift.metrics import latitude_weighted_metrics
from climate_shift.models import FNO2d
from climate_shift.unet import UNet2d

REQUIRED_SHA = "ac0a354032836c5de5463b122d26fad2270fd5ba2ebb48f43074105c8ce7b914"


def verify_historical_only(npz: str | Path):
    path = Path(npz)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REQUIRED_SHA:
        raise ValueError("Frozen v4.5 Historical NPZ SHA256 mismatch")
    x,y,years = load_npz_pairs(path)
    with np.load(path, allow_pickle=False) as z:
        source_times = z["source_times"]
        target_times = z["target_times"]
        latitude = z["latitude"]
    if (x.shape != (1140,1,26,27) or y.shape != x.shape
            or len(years) != 1140 or years.min() != 1980 or years.max() != 2014):
        raise ValueError("Invalid Historical-only input dimensions or years")
    train = years <= 2004
    val = years >= 2005
    if train.sum() != 900 or val.sum() != 240:
        raise ValueError("Invalid frozen Historical train/validation population")
    if not (np.all(target_times-source_times == np.timedelta64(24,"h"))
            and target_times[train].max() < source_times[val].min()):
        raise ValueError("Source-target lead or train/validation boundary violation")
    month = target_times.astype("datetime64[M]").astype(int) % 12 + 1
    if any(int(np.sum(train & (month == m))) != 75 for m in range(1,13)):
        raise ValueError("Historical training does not have 75 samples per month")
    cutoffs, reference = fit_local_monthly_p95(y[train],target_times[train],
                                               min_samples_per_month=20)
    return x,y,years,latitude,target_times,train,val,cutoffs,reference,digest


def fit_historical(npz, output_dir, seed=11, epochs=2, width=8, modes=4,
                   tail_weight=0.0, batch_size=8):
    import torch
    if (epochs<1 or epochs>100 or width!=8 or modes!=4
            or batch_size<1 or tail_weight not in (0.0,1.0,3.0)):
        raise ValueError("Unexpected unfrozen smoke/training configuration")
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refuse to overwrite fitted checkpoints or results")
    x,y,years,latitude,times,train,val,p95,reference,digest=verify_historical_only(npz)
    mu = float(y[train].mean())
    sigma = float(y[train].std())
    if sigma<=1e-8: raise ValueError("Near-zero training standard deviation")
    x_tensor=torch.from_numpy((x-mu)/sigma)
    y_tensor=torch.from_numpy((y-mu)/sigma)
    ids_train=torch.from_numpy(np.flatnonzero(train))
    ids_val=torch.from_numpy(np.flatnonzero(val))
    p95_standard=torch.from_numpy(((p95-mu)/sigma).astype(np.float32))
    month_indices=torch.from_numpy((times.astype("datetime64[M]").astype(int)%12).astype(np.int64))
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False) if not out.exists() else None
    summary={
        "status":"historical_only_checkpoint_fit_no_SSP_or_future_forecast",
        "npz_sha256":digest,"seed":seed,"epochs":epochs,
        "tail_weight":tail_weight,"models":{},
        "training_years":[1980,2004],"validation_years":[2005,2014],
        "train_count":900,"validation_count":240,
        "train_only_normalization":{"mean_K":mu,"sd_K":sigma},
        "local_p95_reference":reference,
        "validation_persistence":latitude_weighted_metrics(x[val],y[val],latitude),
        "ssp_data_read":False,
    }
    for name in ("fno","unet"):
        torch.manual_seed(seed)
        model=(FNO2d(width=width,modes_y=modes,modes_x=modes,depth=2)
               if name=="fno" else UNet2d(width=width))
        optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
        history=[]
        best_loss=float("inf")
        best_state=None
        best_epoch=None
        for epoch in range(1,epochs+1):
            model.train()
            order=ids_train[torch.randperm(len(ids_train))]
            running=0.0
            for b in order.split(batch_size):
                optimizer.zero_grad()
                forecast=model(x_tensor[b])
                squared=(forecast-y_tensor[b]).square()
                if tail_weight:
                    exceed=y_tensor[b,0]>p95_standard[month_indices[b]]
                    weights=1+tail_weight*exceed[:,None].to(squared.dtype)
                    loss=(squared*weights).mean()
                else:
                    loss=squared.mean()
                if not torch.isfinite(loss):
                    raise ValueError(f"{name} nonfinite train loss")
                loss.backward()
                optimizer.step()
                running+=float(loss.detach())*len(b)
            model.eval()
            with torch.no_grad():
                total=0.0
                for b in ids_val.split(32):
                    total+=float((model(x_tensor[b])-y_tensor[b]).square().sum())
                val_loss=total/len(ids_val)/np.prod(y.shape[1:])
            if not np.isfinite(val_loss):
                raise ValueError(f"{name} nonfinite validation MSE")
            history.append({"epoch":epoch,
                            "training_loss_standardized":running/len(ids_train),
                            "validation_mse_standardized":val_loss})
            if val_loss < best_loss:
                best_loss=val_loss
                best_epoch=epoch
                best_state={key:value.detach().clone() for key,value in model.state_dict().items()}
        model.load_state_dict(best_state)
        ckpt=out/f"{name}_seed{seed}_weight{int(tail_weight)}.pt"
        torch.save({"state_dict":best_state,"architecture":name,"seed":seed,
                    "tail_weight":tail_weight,"best_epoch":best_epoch,
                    "normalization_mean_K":mu,"normalization_sd_K":sigma,
                    "training_npz_sha256":digest,"train_year_end":2004,
                    "validation_year_end":2014,"training_only_monthly_p95_K":p95.astype(np.float32),
                    "model_width":width,"fno_modes":modes},ckpt)
        # Reload the exact checkpoint and reproduce outputs before reporting skill.
        saved=torch.load(ckpt,map_location="cpu",weights_only=True)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        with torch.no_grad():
            pred=np.concatenate([model(x_tensor[b]).numpy() for b in ids_val.split(32)],axis=0)*sigma+mu
        report={"checkpoint":ckpt.name,"checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                "parameter_count":sum(p.numel() for p in model.parameters()),
                "best_epoch":best_epoch,"best_validation_mse_standardized":best_loss,
                "validation_weighted":latitude_weighted_metrics(pred,y[val],latitude),
                "validation_local_p95":local_monthly_hot_diagnostics(pred,y[val],times[val],p95,latitude),
                "history":history}
        summary["models"][name]=report
    (out/"historical_fit_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--seed",type=int,default=11)
    p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--tail-weight",type=float,default=0.0)
    args=p.parse_args()
    r=fit_historical(args.input,args.output_dir,seed=args.seed,epochs=args.epochs,
                     tail_weight=args.tail_weight)
    print(json.dumps({"status":r["status"],"seed":r["seed"],"epochs":r["epochs"],
                      "models":{k:{"best_epoch":v["best_epoch"],
                                   "weighted_rmse_K":v["validation_weighted"]["rmse"]}
                                for k,v in r["models"].items()}},indent=2))


if __name__=="__main__":
    main()
