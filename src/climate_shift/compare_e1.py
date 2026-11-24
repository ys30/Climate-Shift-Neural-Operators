"""Matched-input FNO, U-Net and persistence developmental comparison."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .data import load_npz_pairs, chronological_masks
from .split_isolation import audit_split_isolation
from .metrics import regression_metrics, latitude_weighted_metrics
from .models import FNO2d
from .unet import UNet2d
from .extremes import fit_hot_threshold, hot_diagnostics
from .seasonal_extremes import fit_local_monthly_p95, local_monthly_hot_diagnostics
from scripts.audit_model_parameters import parameter_report

def compare(npz: str | Path, output: str | Path, train_end: int, val_end: int,
            test_end: int, embargo_years: int = 0, epochs: int = 4,
            batch_size: int = 8, seed: int = 2026, width: int = 8,
            modes: int = 4, lr: float = 0.001,
            require_pair_times: bool = False) -> dict:
    if epochs < 1 or batch_size < 1 or width < 1 or modes < 1 or lr <= 0:
        raise ValueError("Invalid training hyperparameters")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    x, y, years = load_npz_pairs(npz)
    train, val, test = chronological_masks(years, train_end, val_end, test_end, embargo_years)
    split_audit = audit_split_isolation(npz, (train, val, test), require_pair_times=require_pair_times)
    if x.shape[1] != 1 or y.shape[1] != 1:
        raise ValueError("This comparison requires single-channel temperature data")
    if x.shape[-2] < 2 * modes or x.shape[-1] // 2 + 1 < modes:
        raise ValueError("Grid too small for FNO modes")
    mean, std = float(y[train].mean()), float(y[train].std())
    if std < 1e-8:
        raise ValueError("Training target has near-zero standard deviation")
    xx = torch.from_numpy((x - mean) / std)
    yy = torch.from_numpy((y - mean) / std)
    ids = [torch.from_numpy(np.flatnonzero(m)) for m in (train, val, test)]
    with np.load(npz, allow_pickle=False) as data:
        latitude = data["latitude"] if "latitude" in data.files else None
        target_times = data["target_times"] if "target_times" in data.files else None
    weighted_results = {} if latitude is not None else None
    results = {"persistence": regression_metrics(x[test], y[test])}
    if latitude is not None:
        weighted_results["persistence"] = latitude_weighted_metrics(x[test], y[test], latitude)
    threshold = fit_hot_threshold(y[train], 0.95)
    tail_results = {"persistence": hot_diagnostics(x[test], y[test], threshold, latitude=latitude)}
    seasonal_reference = {"status": "unavailable_missing_target_times"}
    seasonal_tail = None
    if target_times is not None:
        month_ids = target_times[train].astype("datetime64[M]").astype("int64") % 12 + 1
        train_month_counts = {f"{m:02d}": int((month_ids == m).sum()) for m in range(1, 13)}
        if min(train_month_counts.values()) >= 20:
            # Threshold fit sees only training targets; validation/test held out.
            train_cutoffs, seasonal_reference = fit_local_monthly_p95(
                y[train], target_times[train], min_samples_per_month=20
            )
            seasonal_reference["status"] = "available_training_only"
            seasonal_tail = {
                "persistence": local_monthly_hot_diagnostics(
                    x[test], y[test], target_times[test], train_cutoffs, latitude=latitude
                )
            }
        else:
            seasonal_reference = {
                "status": "unavailable_insufficient_training_monthly_coverage",
                "samples_per_month": train_month_counts,
                "minimum_required_per_month": 20,
            }
    diagnostics = {}
    torch.set_num_threads(1)
    for name in ("fno", "unet"):
        torch.manual_seed(seed)
        model = (FNO2d(width=width, modes_y=modes, modes_x=modes, depth=2)
                 if name == "fno" else UNet2d(width=width))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best, best_state, best_epoch = float("inf"), None, None
        epoch_history = []
        for epoch in range(1, epochs + 1):
            model.train()
            order = ids[0][torch.randperm(len(ids[0]))]
            total_train_loss, total_train_seen = 0.0, 0
            for sub in order.split(batch_size):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(model(xx[sub]), yy[sub])
                loss.backward()
                optimizer.step()
                total_train_loss += float(loss.detach()) * len(sub)
                total_train_seen += len(sub)
            model.eval()
            with torch.no_grad():
                val_loss = float(torch.nn.functional.mse_loss(model(xx[ids[1]]), yy[ids[1]]))
            epoch_history.append({
                "epoch": epoch,
                "minibatch_train_mse_standardized": total_train_loss / total_train_seen,
                "validation_mse_standardized": val_loss,
            })
            if val_loss < best:
                best = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds = model(xx[ids[2]]).numpy() * std + mean
        results[name] = regression_metrics(preds, y[test])
        tail_results[name] = hot_diagnostics(preds, y[test], threshold, latitude=latitude)
        if latitude is not None:
            weighted_results[name] = latitude_weighted_metrics(preds, y[test], latitude)
        if seasonal_tail is not None:
            seasonal_tail[name] = local_monthly_hot_diagnostics(
                preds, y[test], target_times[test], train_cutoffs, latitude=latitude
            )
        diagnostics[name] = {
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_accounting": parameter_report(model),
            "best_val_mse_standardized": best,
            "best_epoch": best_epoch,
            "history": epoch_history,
            "minibatch_loss_caveat": "Training batch MSE is recorded during weight updates, not a frozen post-epoch train-set loss",
        }
    report = {
        "status": "development-only; NOT evidence of climate-shift skill",
        "input_sha256": hashlib.sha256(Path(npz).read_bytes()).hexdigest(),
        "year_boundaries": {"train_end": train_end, "val_end": val_end,
                            "test_end": test_end, "embargo_years": embargo_years},
        "split_isolation": split_audit,
        "n_samples": dict(zip(("train", "validation", "test"),
                              (int(m.sum()) for m in (train, val, test)))),
        "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "seed": seed,
                            "width": width, "modes": modes, "lr": lr},
        "train_target_normalization": {"mean": mean, "std": std},
        "diagnostics": diagnostics, "metrics": results,
        "latitude_weighted_metrics": weighted_results,
        "area_weighting_status": "cos(latitude)" if latitude is not None else "missing_latitude_legacy_unweighted",
        "training_reference_p95_threshold": threshold,
        "heat_exceedance_diagnostics": tail_results,
        "seasonal_local_p95_definition": seasonal_reference,
        "seasonal_local_hot_diagnostics": seasonal_tail,
        "limitations": ["Some metrics unweighted; weighted metrics reported separately when latitude exists", "No climate-shift test",
                        "Same epoch count does not imply matched compute",
                        "No physical consistency diagnostics"]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    def sanitize(value):
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    report = sanitize(report)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    for key in ("train-end", "val-end", "test-end"):
        p.add_argument("--" + key, type=int, required=True)
    p.add_argument("--embargo-years", type=int, default=0)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--modes", type=int, default=4)
    p.add_argument("--require-pair-times", action="store_true")
    a = p.parse_args()
    print(json.dumps(compare(a.input, a.output, a.train_end, a.val_end, a.test_end,
                             a.embargo_years, a.epochs, a.batch_size,
                             a.seed, a.width, a.modes, require_pair_times=a.require_pair_times), indent=2))

if __name__ == "__main__":
    main()
