"""Train E1 FNO with predeclared multi-year splits. Development baseline only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import chronological_masks, load_npz_pairs
from .split_isolation import audit_split_isolation
from .metrics import regression_metrics
from .models import FNO2d

def run(npz: str | Path, train_end: int, val_end: int, test_end: int,
        output: str | Path, epochs: int = 10, seed: int = 2026,
        embargo_years: int = 0, width: int = 8, modes: int = 4,
        batch_size: int = 8, lr: float = 0.001) -> dict:
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs/batch_size/lr must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    x, y, years = load_npz_pairs(npz)
    train_mask, val_mask, test_mask = chronological_masks(
        years, train_end, val_end, test_end, embargo_years)
    split_audit = audit_split_isolation(npz, (train_mask, val_mask, test_mask))
    if x.shape[1] != y.shape[1] or x.shape[1] != 1:
        raise ValueError("Restricted to same-variable, single-channel prediction")
    if x.shape[2] < 2 * modes or x.shape[3] // 2 + 1 < modes:
        raise ValueError("Grid too small for FNO modes")
    center = float(y[train_mask].mean())
    scale = float(y[train_mask].std())
    if scale < 1e-10:
        raise ValueError("Training targets have zero/nearly zero variance")
    xt = torch.from_numpy(((x - center) / scale).astype("float32"))
    yt = torch.from_numpy(((y - center) / scale).astype("float32"))
    train_idx = torch.from_numpy(np.flatnonzero(train_mask))
    val_idx = torch.from_numpy(np.flatnonzero(val_mask))
    test_idx = torch.from_numpy(np.flatnonzero(test_mask))
    model = FNO2d(in_channels=1, out_channels=1, width=width,
                  modes_y=modes, modes_x=modes, depth=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_weights = float("inf"), None
    for _ in range(epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        for index in order.split(batch_size):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xt[index]), yt[index])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(model(xt[val_idx]), yt[val_idx]))
        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        predicted = model(xt[test_idx]).numpy() * scale + center
    target = y[test_mask]
    result = {
        "scientific_status": "development baseline; no OOD or future-climate claim",
        "data_sha256": hashlib.sha256(Path(npz).read_bytes()).hexdigest(),
        "configuration": {"train_end": train_end, "val_end": val_end,
                          "test_end": test_end, "embargo_years": embargo_years,
                          "epochs": epochs, "seed": seed, "width": width,
                          "modes": modes, "batch_size": batch_size, "lr": lr},
        "split_isolation": split_audit,
        "sample_counts": {"train": int(train_mask.sum()),
                          "validation": int(val_mask.sum()), "test": int(test_mask.sum())},
        "normalization": {"target_mean_train_only": center,
                          "target_std_train_only": scale},
        "validation_best_mse_standardized": best_val,
        "test": {"FNO": regression_metrics(predicted, target),
                 "persistence": regression_metrics(x[test_mask], target)},
        "metric_units": "same physical units as input source",
        "limitations": ["Unweighted spatial averages", "Test never used for model selection",
                        "Single-channel only", "No climate-shift attribution"]
    }
    dest = Path(output)
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-end", type=int, required=True)
    p.add_argument("--val-end", type=int, required=True)
    p.add_argument("--test-end", type=int, required=True)
    p.add_argument("--embargo-years", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--modes", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    print(json.dumps(run(a.input, a.train_end, a.val_end, a.test_end,
                         a.output, a.epochs, a.seed, a.embargo_years,
                         a.width, a.modes, a.batch_size), indent=2))

if __name__ == "__main__":
    main()
