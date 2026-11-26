"""Run a frozen three-seed Historical-only CMIP6 train/validation loss arm.

Runs exact saved Historical fit code; does not access SSP Zarr or future inputs.
Stores all model checkpoints and all arms, regardless of their performance.
Historical validation is NOT an independent prospective test set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .fit_cmip6_awi_historical import fit_historical, REQUIRED_SHA
except ImportError:
    from fit_cmip6_awi_historical import fit_historical, REQUIRED_SHA

SEEDS = (11, 22, 33)
WEIGHTS = (0.0, 1.0, 3.0)
EPOCHS = 100
MODELS = ("fno", "unet")


def aggregate(reports, tail_weight):
    if tail_weight not in WEIGHTS or len(reports) != len(SEEDS):
        raise ValueError("Unfrozen arm configuration")
    if [r["seed"] for r in reports] != list(SEEDS):
        raise ValueError("Missing or changed independent seeds")
    if any(r["npz_sha256"] != REQUIRED_SHA or r["tail_weight"] != tail_weight or
           r["epochs"] != EPOCHS or r["ssp_data_read"] is not False for r in reports):
        raise ValueError("Historical dataset, fixed run budget or SSP isolation changed")
    metrics = {}
    for model in MODELS:
        metrics[model] = {}
        for label, path in (
            ("weighted_rmse_K", ("validation_weighted", "rmse")),
            ("weighted_mae_K", ("validation_weighted", "mae")),
            ("weighted_bias_K", ("validation_weighted", "bias")),
            ("local_p95_recall", ("validation_local_p95", "hot_recall")),
            ("local_p95_precision", ("validation_local_p95", "hot_precision")),
            ("predicted_hot_fraction", ("validation_local_p95", "predicted_hot_fraction")),
        ):
            values = [r["models"][model][path[0]][path[1]] for r in reports]
            if not all(v is not None and np.isfinite(v) for v in values):
                raise ValueError("Incomplete or nonfinite model validation metrics")
            metrics[model][label] = {
                "individual": [float(x) for x in values],
                "mean": float(np.mean(values)),
                "sd_across_initialization_seeds": float(np.std(values, ddof=1)),
            }
        metrics[model]["parameter_count_by_seed"] = [
            r["models"][model]["parameter_count"] for r in reports
        ]
        metrics[model]["best_epoch_by_seed"] = [
            r["models"][model]["best_epoch"] for r in reports
        ]
        metrics[model]["checkpoints"] = [
            {"seed":r["seed"],
             "file":f"seed_{r['seed']}/{r['models'][model]['checkpoint']}",
             "sha256":r["models"][model]["checkpoint_sha256"]}
            for r in reports
        ]
        if len(set(metrics[model]["parameter_count_by_seed"])) != 1:
            raise ValueError("Model architecture parameters changed across seeds")
    baseline = reports[0]["validation_persistence"]
    if any(r["validation_persistence"] != baseline for r in reports):
        raise ValueError("Unmatched Persistence baseline across seeds")
    return {
        "status": "frozen_cmip6_historical_validation_only_no_ssp_evaluation",
        "data_sha256":REQUIRED_SHA,
        "seed_ids":list(SEEDS),
        "tail_weight":tail_weight,
        "training_hot_multiplier":1+tail_weight,
        "epochs":EPOCHS,
        "train_cohort":"AWI CMIP6 Historical 1980–2004, 900 sparse daily pairs",
        "validation_cohort":"AWI CMIP6 Historical 2005–2014, 240 sparse daily pairs",
        "source_variable":"native daily mean tas Kelvin, one-native-day lead",
        "validation_persistence":baseline,
        "models":metrics,
        "ssp_data_read":False,
        "selection_policy":"Checkpoint per seed/arm selected ONLY by Historical unweighted validation MSE",
        "cautions":[
            "Historical 2005–2014 validation is used for checkpoint selection, not an unbiased new test",
            "Three random initialization seeds do not substitute for independent climate model members",
            "Historical sparse training days are not continuous heatwave events",
            "Do not select weight and then claim unbiased heldout validation improvement",
            "No SSP245 or SSP585 forecast or future warming-signal fidelity has been evaluated",
        ],
    }


def run(npz: str, output: str, weight: float):
    if weight not in WEIGHTS:
        raise ValueError("Weights restricted to predeclared 0/1/3")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Do not overwrite prior Historical fitted results")
    reports = []
    for seed in SEEDS:
        run_dir = root / f"seed_{seed}"
        reports.append(fit_historical(npz, run_dir, seed=seed, epochs=EPOCHS,
                                      tail_weight=weight))
        if reports[-1]["seed"] != seed:
            raise ValueError("Seed provenance mismatch")
    report = aggregate(reports,weight)
    (root/"arm_summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--tail-weight",type=float,required=True,choices=WEIGHTS)
    a=p.parse_args()
    r=run(a.input,a.output_dir,a.tail_weight)
    print(json.dumps({
        "status":r["status"],
        "weight":r["tail_weight"],
        "seeds":r["seed_ids"],
        "epochs":r["epochs"],
        "validation_persistence_rmse_K":r["validation_persistence"]["rmse"],
        "models":{name:{"weighted_rmse_K":d["weighted_rmse_K"]["mean"],
                         "local_p95_recall":d["local_p95_recall"]["mean"],
                         "local_p95_precision":d["local_p95_precision"]["mean"]}
                  for name,d in r["models"].items()},
    },indent=2))


if __name__=="__main__":
    main()
