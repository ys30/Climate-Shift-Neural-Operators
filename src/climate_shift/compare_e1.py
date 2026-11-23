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
from .metrics import regression_metrics
from .models import FNO2d
from .unet import UNet2d
from .extremes import fit_hot_threshold, hot_diagnostics

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
    results = {"persistence": regression_metrics(x[test], y[test])}
    threshold = fit_hot_threshold(y[train], 0.95)
    tail_results = {"persistence": hot_diagnostics(x[test], y[test], threshold)}
    diagnostics = {}
    torch.set_num_threads(1)
    for name in ("fno", "unet"):
        torch.manual_seed(seed)
        model = (FNO2d(width=width, modes_y=modes, modes_x=modes, depth=2)
                 if name == "fno" else UNet2d(width=width))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best, best_state = float("inf"), None
        for _ in range(epochs):
            model.train()
            order = ids[0][torch.randperm(len(ids[0]))]
            for sub in order.split(batch_size):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(model(xx[sub]), yy[sub])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(torch.nn.functional.mse_loss(model(xx[ids[1]]), yy[ids[1]]))
            if val_loss < best:
                best = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds = model(xx[ids[2]]).numpy() * std + mean
        results[name] = regression_metrics(preds, y[test])
        tail_results[name] = hot_diagnostics(preds, y[test], threshold)
        diagnostics[name] = {"parameter_count": sum(p.numel() for p in model.parameters()),
                             "best_val_mse_standardized": best}
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
        "training_reference_p95_threshold": threshold,
        "heat_exceedance_diagnostics": tail_results,
        "limitations": ["Unweighted grid metrics", "No climate-shift test",
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
