"""Run reproducible E1 development comparisons across independent seeds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from climate_shift.compare_e1 import compare


def run_seeds(input_path: str, output_dir: str, train_end: int, val_end: int,
              test_end: int, seeds: tuple[int, ...] = (11, 22, 33),
              epochs: int = 4, width: int = 8, modes: int = 4,
              require_pair_times: bool = False):
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("Supply at least two unique seeds")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory must be empty or nonexistent")
    reports = []
    for seed in seeds:
        reports.append(compare(input_path, out / f"seed_{seed}.json", train_end,
                               val_end, test_end, epochs=epochs,
                               seed=seed, width=width, modes=modes,
                               require_pair_times=require_pair_times))
    if require_pair_times and any(r["split_isolation"]["status"] != "passed" for r in reports):
        raise ValueError("Every seed must pass a strict timestamp isolation audit")
    summary = {"status": "development only; no scientific inference",
               "input_sha256": reports[0]["input_sha256"],
               "split_isolation": reports[0]["split_isolation"],
               "seeds": list(seeds), "models": {}}
    for name in ("fno", "unet", "persistence"):
        summary["models"][name] = {}
        for metric in ("mae", "rmse", "bias"):
            values = np.array([r["metrics"][name][metric] for r in reports])
            summary["models"][name][metric] = {
                "mean": float(values.mean()),
                "sd_across_seeds": float(values.std(ddof=1)),
                "individual": values.tolist(),
            }
    for name in ("fno", "unet", "persistence"):
        summary["models"][name]["heat_recall"] = {
            "individual": [r["heat_exceedance_diagnostics"][name]["hot_recall"] for r in reports]
        }
    (out / "seed_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-end", type=int, required=True)
    p.add_argument("--val-end", type=int, required=True)
    p.add_argument("--test-end", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--modes", type=int, default=4)
    p.add_argument("--require-pair-times", action="store_true")
    a = p.parse_args()
    print(json.dumps(run_seeds(a.input, a.output_dir, a.train_end, a.val_end,
                               a.test_end, tuple(a.seeds), a.epochs, a.width, a.modes,
                               require_pair_times=a.require_pair_times),
                     indent=2))


if __name__ == "__main__":
    main()
