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
              require_pair_times: bool = False, tail_weight: float = 0.0):
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
                               require_pair_times=require_pair_times,
                               tail_weight=tail_weight))
    if require_pair_times and any(r["split_isolation"]["status"] != "passed" for r in reports):
        raise ValueError("Every seed must pass a strict timestamp isolation audit")
    if len({r["input_sha256"] for r in reports}) != 1:
        raise ValueError("All seeds must use the identical input dataset checksum")
    weighted_availability = [r.get("latitude_weighted_metrics") is not None for r in reports]
    if any(weighted_availability) and not all(weighted_availability):
        raise ValueError("Area-weighted scores cannot be mixed across seeds")
    summary = {"status": "development only; no scientific inference",
               "input_sha256": reports[0]["input_sha256"],
               "split_isolation": reports[0]["split_isolation"],
               "seeds": list(seeds), "tail_weight": tail_weight, "models": {}}
    for name in ("fno", "unet", "persistence"):
        summary["models"][name] = {}
        for metric in ("mae", "rmse", "bias"):
            values = np.array([r["metrics"][name][metric] for r in reports])
            summary["models"][name][metric] = {
                "mean": float(values.mean()),
                "sd_across_seeds": float(values.std(ddof=1)),
                "individual": values.tolist(),
            }
    if all(r.get("latitude_weighted_metrics") is not None for r in reports):
        for name in ("fno", "unet", "persistence"):
            for metric in ("mae", "rmse", "bias"):
                values = np.array([r["latitude_weighted_metrics"][name][metric] for r in reports])
                summary["models"][name]["weighted_" + metric] = {
                    "mean": float(values.mean()),
                    "sd_across_seeds": float(values.std(ddof=1)),
                    "individual": values.tolist(),
                }
    for name in ("fno", "unet", "persistence"):
        recall = np.asarray(
            [r["heat_exceedance_diagnostics"][name]["hot_recall"] for r in reports], dtype=float
        )
        summary["models"][name]["heat_recall"] = {
            "individual": [float(v) if np.isfinite(v) else None for v in recall],
            "mean": float(recall.mean()) if np.isfinite(recall).all() else None,
            "sd_across_seeds": float(recall.std(ddof=1)) if np.isfinite(recall).all() else None,
        }
    summary["seasonal_local_p95_definition"] = reports[0].get("seasonal_local_p95_definition")
    seasonal_available = all(r.get("seasonal_local_hot_diagnostics") is not None for r in reports)
    if seasonal_available:
        for name in ("fno", "unet", "persistence"):
            values = [r["seasonal_local_hot_diagnostics"][name]["hot_recall"] for r in reports]
            good = all(v is not None and np.isfinite(v) for v in values)
            summary["models"][name]["monthly_local_hot_recall"] = {
                "individual": values,
                "mean": float(np.mean(values)) if good else None,
                "sd_across_seeds": float(np.std(values, ddof=1)) if good else None,
            }
    # Aggregate independent heldout-year diagnostics across seeds. All years
    # must be identical across runs, and model scores are never refitted here.
    if all(r.get("per_test_year_metrics") is not None for r in reports):
        years = sorted(reports[0]["per_test_year_metrics"]["persistence"])
        for r in reports:
            if any(sorted(r["per_test_year_metrics"][name]) != years
                   for name in ("fno", "unet", "persistence")):
                raise ValueError("Heldout year coverage differs across seeds")
        summary["heldout_years"] = years
        summary["per_test_year_models"] = {}
        for year in years:
            summary["per_test_year_models"][year] = {}
            for name in ("fno", "unet", "persistence"):
                items = [r["per_test_year_metrics"][name][year] for r in reports]
                if len({item["n_samples"] for item in items}) != 1:
                    raise ValueError("Heldout annual sample counts differ across seeds")
                entry = {"n_samples": items[0]["n_samples"]}
                for key, source in (("area_weighted_rmse", ("area_weighted", "rmse")),
                                    ("area_weighted_mae", ("area_weighted", "mae")),
                                    ("monthly_local_hot_recall", ("monthly_local_hot", "hot_recall"))):
                    values = [item[source[0]][source[1]] if item.get(source[0]) is not None else None
                              for item in items]
                    good = all(v is not None and np.isfinite(v) for v in values)
                    entry[key] = {
                        "mean": float(np.mean(values)) if good else None,
                        "sd_across_seeds": float(np.std(values, ddof=1)) if good else None,
                        "individual": values,
                    }
                summary["per_test_year_models"][year][name] = entry
    summary["area_weighting_status"] = (
        "cos(latitude) in weighted metrics and hot-tail diagnostics"
        if all(weighted_availability) else "unweighted legacy source"
    )
    summary["paired_differences"] = {}
    for newer, older in (("fno", "persistence"), ("unet", "persistence"), ("fno", "unet")):
        pair = f"{newer}_minus_{older}"
        summary["paired_differences"][pair] = {}
        for metric in ("rmse", "mae"):
            delta = np.asarray(
                [r["metrics"][newer][metric] - r["metrics"][older][metric] for r in reports],
                dtype=float,
            )
            summary["paired_differences"][pair][metric] = {
                "individual": delta.tolist(), "mean": float(delta.mean()),
                "sd_across_seeds": float(delta.std(ddof=1)),
            }
            if all(weighted_availability):
                weighted_delta = np.asarray(
                    [r["latitude_weighted_metrics"][newer][metric] -
                     r["latitude_weighted_metrics"][older][metric] for r in reports], dtype=float,
                )
                summary["paired_differences"][pair]["weighted_" + metric] = {
                    "individual": weighted_delta.tolist(), "mean": float(weighted_delta.mean()),
                    "sd_across_seeds": float(weighted_delta.std(ddof=1)),
                }
        p_recall = summary["models"][newer]["heat_recall"]["individual"]
        r_recall = summary["models"][older]["heat_recall"]["individual"]
        if all(v is not None for v in p_recall + r_recall):
            recall_delta = np.asarray(p_recall) - np.asarray(r_recall)
            summary["paired_differences"][pair]["heat_recall"] = {
                "individual": recall_delta.tolist(), "mean": float(recall_delta.mean()),
                "sd_across_seeds": float(recall_delta.std(ddof=1)),
            }
        else:
            summary["paired_differences"][pair]["heat_recall"] = {
                "individual": None, "mean": None, "sd_across_seeds": None,
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
    p.add_argument("--tail-weight", type=float, default=0.0)
    a = p.parse_args()
    print(json.dumps(run_seeds(a.input, a.output_dir, a.train_end, a.val_end,
                               a.test_end, tuple(a.seeds), a.epochs, a.width, a.modes,
                               require_pair_times=a.require_pair_times,
                               tail_weight=a.tail_weight),
                     indent=2))


if __name__ == "__main__":
    main()
