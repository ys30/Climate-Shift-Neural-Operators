"""v5.6: audit frozen v5.5 GCM-year contrasts; descriptive, never ensemble inference."""
import argparse
import json
import math
from pathlib import Path

GCMS = ("AWI", "BCC")
SSPS = ("ssp245", "ssp585")
SCOPES = ("all", "2030", "2050", "2080")
MODELS = ("fno", "unet")
WEIGHTS = (1, 2, 4)
FIELDS = ("weighted_rmse_K", "persistence_weighted_rmse_K",
          "rmse_improvement_fraction_vs_own_persistence", "local_p95_recall",
          "persistence_local_p95_recall", "recall_delta_vs_own_persistence",
          "local_p95_precision")

def audit(report):
    if report.get("status") != "paired_within_GCM_projected_temperature_forecast_contrasts_no_pooled_significance":
        raise ValueError("Only verified v5.5 frozen comparisons accepted")
    rows = report.get("rows", [])
    expected = {(g, s, y, m, w) for g in GCMS for s in SSPS
                for y in SCOPES for m in MODELS for w in WEIGHTS}
    indexed = {}
    for r in rows:
        key = (r["source_gcm"], r["ssp"], r["scope"], r["model"],
               r["training_hot_multiplier"])
        if key in indexed:
            raise ValueError("Duplicated source/scenario/year/model/weight")
        if key not in expected:
            raise ValueError("Unexpected contrast stratum")
        for field in FIELDS:
            v = r[field]
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                raise ValueError("Missing or non-finite score: " + field)
        if r["n_source_target_pairs"] != (72 if r["scope"] == "all" else 24):
            raise ValueError("Unfrozen date-count / sampling scope")
        if r["persistence_weighted_rmse_K"] <= 0 or r["weighted_rmse_K"] < 0:
            raise ValueError("Invalid RMSE")
        if abs((1 - r["weighted_rmse_K"] / r["persistence_weighted_rmse_K"])
               - r["rmse_improvement_fraction_vs_own_persistence"]) > 1e-8:
            raise ValueError("Inconsistent native persistence contrast")
        if abs((r["local_p95_recall"] - r["persistence_local_p95_recall"])
               - r["recall_delta_vs_own_persistence"]) > 1e-8:
            raise ValueError("Inconsistent recall contrast")
        for f in ("local_p95_recall", "persistence_local_p95_recall", "local_p95_precision"):
            if not 0 <= r[f] <= 1:
                raise ValueError("Out-of-range probability: " + f)
        indexed[key] = r
    if set(indexed) != expected:
        raise ValueError("Incomplete 96-cell comparison: missing " + str(len(expected - set(indexed))))
    year_rows = []
    for g in GCMS:
        for s in SSPS:
            for year in SCOPES[1:]:
                base = indexed[(g, s, year, "fno", 1)]
                tail = indexed[(g, s, year, "fno", 4)]
                year_rows.append({
                    "gcm": g, "ssp": s, "year": int(year),
                    "fno1_rmse_improvement_pct": round(100 * base["rmse_improvement_fraction_vs_own_persistence"], 4),
                    "fno4_minus_fno1_recall_pp": round(100 * (tail["local_p95_recall"] - base["local_p95_recall"]), 4),
                    "fno4_minus_fno1_rmse_K": round(tail["weighted_rmse_K"] - base["weighted_rmse_K"], 6),
                    "persistence_recall_exceeds_fno1": base["persistence_local_p95_recall"] > base["local_p95_recall"]
                })
    aggregate = []
    for g in GCMS:
        for s in SSPS:
            one = indexed[(g, s, "all", "fno", 1)]
            four = indexed[(g, s, "all", "fno", 4)]
            aggregate.append({
                "gcm":g, "ssp":s,
                "fno1_rmse_improvement_pct":round(100 * one["rmse_improvement_fraction_vs_own_persistence"],4),
                "tail4_recall_delta_pp":round(100*(four["local_p95_recall"]-one["local_p95_recall"]),4),
                "tail4_rmse_delta_K":round(four["weighted_rmse_K"]-one["weighted_rmse_K"],6),
                "persistence_recall_exceeds_fno4":one["persistence_local_p95_recall"]>four["local_p95_recall"]
            })
    return {
        "status":"verified_v56_descriptive_robustness_no_significance",
        "parent":"frozen_v55_96_contrasts",
        "independent_gcm_count":2, "members_per_gcm":1,
        "year_level_cells":year_rows, "scenario_aggregate_cells":aggregate,
        "direction_counts":{
            "fno1_rmse_better_years":sum(v["fno1_rmse_improvement_pct"]>0 for v in year_rows),
            "year_cells":len(year_rows),
            "tail4_recall_up_years":sum(v["fno4_minus_fno1_recall_pp"]>0 for v in year_rows),
            "tail4_rmse_worse_years":sum(v["fno4_minus_fno1_rmse_K"]>0 for v in year_rows)
        },
        "limitations":[
            "Descriptive scenario-year contrasts are correlated, not 12 independent GCM replications",
            "No gridcell, seed, or SSP-year p-values or bootstrap significance permitted",
            "Sparse target dates do not establish heatwave duration or warming-signal fidelity",
            "No SSP fitting, checkpoint selection, new data or parameter changes"
        ]
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    a=parser.parse_args()
    destination=Path(a.output)
    if destination.exists():
        raise FileExistsError("Do not overwrite frozen audit")
    result=audit(json.loads(Path(a.input).read_text()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["direction_counts"], indent=2))

if __name__=="__main__":
    main()
