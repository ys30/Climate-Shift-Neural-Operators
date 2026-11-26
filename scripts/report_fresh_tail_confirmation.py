"""Summarize PREDECLARED tail-loss comparisons on untouched heldout years.

Pure stdlib: consumes existing JSON output, never retrains, retunes or
consults test targets to change loss weight selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("0", "1", "3")
MODELS = ("fno", "unet")
YEARS = ("2015", "2016", "2017")


def summarize(report: dict) -> dict:
    if report.get("status") != "fresh_heldout_confirmation_not_CMIP6_climate_shift":
        raise ValueError("Expected untouched historical confirmation artifact")
    if report.get("candidate_tail_weights") != [0, 1, 3]:
        raise ValueError("Predeclared candidate weights changed")
    if report.get("heldout_years") != [2015, 2016, 2017]:
        raise ValueError("Unexpected heldout years")
    arms = report["arms"]
    if set(arms) != set(ARMS) or len({arms[k]["input_sha256"] for k in ARMS}) != 1:
        raise ValueError("Arms must share one exact ERA5 input checksum")

    rows = []
    for model in MODELS:
        for weight in ARMS:
            item = arms[weight]["models"][model]
            rmse = item["weighted_rmse"]
            recall = item["local_p95_recall"]
            precision = item["local_p95_precision_by_seed"]
            if (len(rmse["individual"]) != 3 or len(recall["individual"]) != 3
                    or len(precision) != 3):
                raise ValueError("Exactly three independent initialization seeds required")
            if set(item["per_year"]) != set(YEARS):
                raise ValueError("Incomplete heldout year scores")
            row = {
                "model": model,
                "train_hot_multiplier": 1 + int(weight),
                "weighted_rmse_K": rmse["mean"],
                "local_p95_recall": recall["mean"],
                "local_p95_precision": sum(precision) / len(precision),
                "yearwise": {},
            }
            for year in YEARS:
                per_year = item["per_year"][year]
                hot_rmse = per_year["hot_z_gt_2_rmse_by_seed"]
                if len(hot_rmse) != 3 or any(x is None for x in hot_rmse):
                    raise ValueError("Missing 2015–2017 extreme-hot conditional RMSE")
                row["yearwise"][year] = {
                    "weighted_rmse_K": per_year["weighted_rmse"]["mean"],
                    "local_p95_recall": per_year["local_p95_recall"]["mean"],
                    "extreme_hot_z_gt_2_rmse_K": sum(hot_rmse) / 3,
                }
            rows.append(row)

    paired = {}
    for model in MODELS:
        control = arms["0"]["models"][model]
        paired[model] = {}
        for weight in ("1", "3"):
            treat = arms[weight]["models"][model]
            # Paired by seed; do not treat the three seeds as climate years.
            deltas = {
                key: [
                    v - u for u, v in zip(control[metric]["individual"],
                                         treat[metric]["individual"])
                ]
                for key, metric in (
                    ("weighted_rmse_K", "weighted_rmse"),
                    ("local_p95_recall", "local_p95_recall"),
                )
            }
            paired[model][weight] = {
                key: {"per_seed": vals, "mean": sum(vals) / 3,
                      "all_seeds_favor_treatment": (
                          all(v < 0 for v in vals) if key == "weighted_rmse_K"
                          else all(v > 0 for v in vals))}
                for key, vals in deltas.items()
            }

    return {
        "status": "locked_historical_confirmation_all_candidate_weights_reported",
        "new_years": list(YEARS),
        "source_data_sha256": report["data_sha256"],
        "arm_rows": rows,
        "paired_against_standard_mse": paired,
        "cautions": [
            "Weight candidates were chosen after exploratory 2012–2014 analysis; test only the 2015–2017 confirmation claim.",
            "A new independent future climate projection is required for CMIP6 SSP shift claims.",
            "Three initialization seeds and three weather years do not justify iid gridcell statistical tests.",
            "Do not retrospectively select a treatment and report its test error as an unbiased performance estimate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    result = summarize(json.loads(Path(args.input).read_text()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
