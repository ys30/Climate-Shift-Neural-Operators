"""E1 v5.27: descriptive three-objective Pareto audit of archived v5.25.

Objectives: minimize native weighted next-day RMSE, maximize same-pixel >=3day
local-P95 event-day recall and precision. No checkpoint tuning, no future reads.
Seed means are descriptive, not independent climate ensembles.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from collections import Counter
from scripts.audit_cmip6_v525_three_gcm_contiguous_events import STATUS as SOURCE_STATUS

STATUS="v527_three_GCM_contiguous_event_multiobjective_descriptive_audit"
GCMS=("AWI","BCC","CAMS")
SCENARIOS=("ssp245","ssp585")

def finite(x):
    if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x):
        raise ValueError("Invalid numerical result in frozen v5.25 archive")
    return float(x)

def triple(row):
    result=(finite(row["rmse_K"]),finite(row["event_day_recall"]),finite(row["event_day_precision"]))
    if result[0]<0 or not 0<=result[1]<=1 or not 0<=result[2]<=1:
        raise ValueError("Unphysical RMSE or event probability")
    return result

def dominates(a,b,eps=1e-9):
    """Lower RMSE and greater Recall, Precision; at least one strict gain."""
    return (a[0]<=b[0]+eps and a[1]>=b[1]-eps and a[2]>=b[2]-eps
            and (a[0]<b[0]-eps or a[1]>b[1]+eps or a[2]>b[2]+eps))

def inspect(source):
    if (source.get("status")!=SOURCE_STATUS or source.get("independent_GCM_count")!=3
        or source.get("scenario_cells")!=6 or source.get("neural_arm_rows")!=36
        or source.get("frozen_checkpoint_count")!=54
        or source.get("ssp_used_for_fitting_or_selection") is not False):
        raise ValueError("Require completed unmodified three-GCM v5.25 report")
    arm=source.get("neural_arms",[])
    keys={(r["gcm"],r["scenario"],r["architecture"],r["historical_tail_weight"]) for r in arm}
    expected={(g,s,m,w) for g in GCMS for s in SCENARIOS for m in ("fno","unet") for w in (0,1,3)}
    if len(arm)!=36 or keys!=expected:
        raise ValueError("Missing or repeated source frozen neural arm")
    return arm

def analyze(source):
    arms=inspect(source)
    rows=[];grouped=[]
    for g in GCMS:
        for s in SCENARIOS:
            records=[r for r in arms if r["gcm"]==g and r["scenario"]==s]
            baseline=records[0]
            # Verify that the same native-model Persistence baseline was reused
            # across all six separate FNO/U-Net frozen arms.
            for r in records:
                for key in ("persistence_rmse_K","persistence_event_day_recall",
                            "persistence_event_day_precision"):
                    if r[key] is None or not math.isclose(finite(r[key]),finite(baseline[key]),abs_tol=1e-9):
                        raise ValueError("Native Persistence not held fixed for all original model arms")
            candidates=[{"gcm":g,"scenario":s,"model":"persistence",
               "loss_multiplier":None,
               "rmse_K":finite(baseline["persistence_rmse_K"]),
               "event_day_recall":baseline["persistence_event_day_recall"],
               "event_day_precision":baseline["persistence_event_day_precision"]}]
            for r in records:
                candidates.append({"gcm":g,"scenario":s,
                  "model":r["architecture"],
                  "loss_multiplier":int(r["training_loss_multiplier"]),
                  "rmse_K":finite(r["model_rmse_K"]),
                  "event_day_recall":r["model_event_day_recall"],
                  "event_day_precision":r["model_event_day_precision"]})
            # Null means that event detection precision/recall is undefined.
            # Do not silently treat missing event scores as zero.
            pts=[triple(c) for c in candidates]
            for i,c in enumerate(candidates):
                who=[candidates[j]["model"]+("" if candidates[j]["loss_multiplier"] is None
                    else f"_{candidates[j]['loss_multiplier']}x") for j in range(len(candidates))
                     if j!=i and dominates(pts[j],pts[i])]
                rows.append(dict(c,
                  "pareto_nondominated":len(who)==0,
                  "dominated_by_count":len(who),
                  "dominators":";".join(who)))
            selected=rows[-7:]
            grouped.append({"gcm":g,"scenario":s,
              "n_candidates":len(selected),"n_pareto":sum(r["pareto_nondominated"] for r in selected),
              "persistence_on_frontier":selected[0]["pareto_nondominated"],
              "fno_1x_on_frontier":next(r["pareto_nondominated"] for r in selected if r["model"]=="fno" and r["loss_multiplier"]==1),
              "fno_4x_on_frontier":next(r["pareto_nondominated"] for r in selected if r["model"]=="fno" and r["loss_multiplier"]==4),
              "any_fno_dominates_persistence":any(dominates(triple(r),pts[0]) for r in selected[1:] if r["model"]=="fno"),
              "any_neural_dominates_persistence":any(dominates(triple(r),pts[0]) for r in selected[1:]),
              "persistence_dominates_any_neural":any(dominates(pts[0],triple(r)) for r in selected[1:]),
              "fno_4x_precision_change_pp":100*(
                 next(r["event_day_precision"] for r in selected if r["model"]=="fno" and r["loss_multiplier"]==4)
                -next(r["event_day_precision"] for r in selected if r["model"]=="fno" and r["loss_multiplier"]==1))})
    if len(rows)!=42 or len(grouped)!=6:raise ValueError("Expected 7 candidate models × 6 SSP cases")
    return {"status":STATUS,"source_run":45948547935,
      "original_neural_checkpoint_count":54,"independent_GCM_count":3,
      "scenario_cells":6,"candidate_rows":42,"no_future_refit":True,
      "dominance_epsilon":1e-9,
      "objectives":{"rmse_K":"minimize","event_day_recall":"maximize",
                    "event_day_precision":"maximize"},
      "scenarios":grouped,"candidate_dominance":rows,
      "descriptive_counts":{
        "GCM_with_both_SSP_persistence_on_frontier":sum(all(x["persistence_on_frontier"] for x in grouped if x["gcm"]==g) for g in GCMS),
        "GCM_with_both_SSP_some_neural_dominates_persistence":sum(all(x["any_neural_dominates_persistence"] for x in grouped if x["gcm"]==g) for g in GCMS),
        "cells_where_fno_4x_is_Pareto_optimal":sum(x["fno_4x_on_frontier"] for x in grouped),
        "cells_where_fno_1x_is_Pareto_optimal":sum(x["fno_1x_on_frontier"] for x in grouped),
        "cells_where_fno_4x_precision_decreases":sum(x["fno_4x_precision_change_pp"]<0 for x in grouped)},
      "limits":[
        "Exploratory Pareto analysis uses SAME 2050 summer date window used to evaluate all model arms; not independent hyperparameter selection",
        "Non-dominance is criterion-dependent (weighted RMSE, event-day Recall, Precision) and NOT probability of better future performance",
        "Two SSP cases paired within each of only three independent GCMs",
        "Six neural loss arms per cell are 3-seed mean metrics, not independently trained new climate GCMs",
        "No environmental impact, observational heatwave, tasmax or event-duration matching inference",
        "No p-values or bootstrapped confidence intervals from dependent spatial-temporal pixels"]}

def export(data,root):
    root=Path(root)
    if root.exists():raise FileExistsError("Refuse to overwrite audit results")
    root.mkdir(parents=True)
    (root/"v527_three_GCM_pareto_event_audit.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    for key,name in (("candidate_dominance","v527_42_candidate_metrics.csv"),
                     ("scenarios","v527_six_SSP_pareto_counts.csv")):
        rows=data[key]
        with (root/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    out=["# E1 v5.27: Three-GCM RMSE / sustained-hot-event Recall / Precision Pareto audit","",
      "An exploratory descriptive comparison of seven options (native Persistence and 2 architectures × 3 pre-existing frozen Historical-only loss arms) in each GCM–SSP cell. A model is dominated only if another is no worse in weighted RMSE, event-day Recall and Precision and strictly better in at least one. Non-dominated does not mean best in every objective or independently validated.","",
      "| GCM | SSP | Nondominated / 7 | Persistence on frontier? | FNO 1× on frontier? | FNO 4× on frontier? | FNO 4×−1× Precision (pp) |",
      "|---|---|---:|---|---|---|---:|"]
    for x in data["scenarios"]:
        out.append(f"| {x['gcm']} | {x['scenario']} | {x['n_pareto']} | {x['persistence_on_frontier']} | {x['fno_1x_on_frontier']} | {x['fno_4x_on_frontier']} | {x['fno_4x_precision_change_pp']:+.2f} |")
    out+=["","Each GCM provides only one member and two correlated SSP scenarios; this is not an inferential sample of six independent GCMs. The exercise evaluates simulated daily mean TAS ≥3-day pixel-local event labels on one 2050 summer window, with window edge censoring. Do not choose a winning training weight from the same test window and claim it was preselected. Instead, Pareto-front results motivate future multi-year blind evaluations.","",
      "Input: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45948547935",""]
    (root/"v527_three_GCM_multiobjective_results.md").write_text("\n".join(out),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output-dir",required=True)
    a=ap.parse_args()
    r=analyze(json.loads(Path(a.input).read_text()))
    export(r,a.output_dir)
    print(json.dumps({"status":r["status"],"counts":r["descriptive_counts"],
      "scenarios":r["scenarios"]},indent=2))
if __name__=="__main__":main()
