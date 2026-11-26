"""E1 v5.13: compare immutable original sparse SSP scores with new 2040 cohort.

All comparisons are source-matched, scenario-matched, and relative to their
own native Persistence. Contrasting distinct dates and years is descriptive.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

SOURCES=("AWI","BCC")
SCENARIOS=("ssp245","ssp585")
MODELS=("fno","unet")
WEIGHTS=(0,1,3)
KEYS=("weighted_rmse_K","local_p95_recall","local_p95_precision")
ORIGINAL_STATUS={"AWI":"first_frozen_historical_to_sparse_SSP_forecast_evaluation",
                 "BCC":"first_independent_GCM_BCC_Historical_to_sparse_SSP_frozen_forecast"}

def finite(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError("Non-finite model metric")
    return float(value)

def old_groups(source,report,ssp):
    if report.get("status")!=ORIGINAL_STATUS[source]:
        raise ValueError("Unverified original SSP report")
    if report.get("future_years")!=[2030,2050,2080] or report.get("future_month_days")!=[8,20]:
        raise ValueError("Changed original SSP schedule")
    if source=="AWI":
        if report.get("ssp_used_for_fitting_or_selection") is not False or len(report.get("checkpoint_sha256",{}))!=18:
            raise ValueError("Unfrozen original AWI models")
        return report["scenarios"][ssp]["groups"]["all"]
    if report.get("ssp_used_for_fitting_or_checkpoint_selection") is not False or len(report.get("checkpoint_sha",{}))!=18:
        raise ValueError("Unfrozen original BCC models")
    return report["scenario_results"][ssp]["scores"]["all"]

def extract_old(source,group,model,weight):
    if group["n_pairs"]!=72:
        raise ValueError("Original cohort must contain 72 pairs")
    original=group["models"][model][str(weight)]
    p=group["persistence"]
    hot="local_p95" if source=="AWI" else "monthly_hot"
    return {
        "rmse":finite(original["weighted_rmse_K"]["mean"]),
        "recall":finite(original["local_p95_recall"]["mean"]),
        "precision":finite(original["local_p95_precision"]["mean"]),
        "persistence_rmse":finite(p["weighted"]["rmse"]),
        "persistence_recall":finite(p[hot]["hot_recall"])
    }

def extract_new(group,model,weight):
    if group["n_pairs"]!=84 or group.get("year")!=2040:
        raise ValueError("New cohort must be complete 2040 test")
    record=group["models"][f"{model}_weight{weight}"]
    p=group["models"]["persistence"]
    for key in ("rmse_K","p95_recall","p95_precision"):
        if len(record[key]["individual"])!=3:
            raise ValueError("Missing three frozen-seed results")
        individual=[finite(v) for v in record[key]["individual"]]
        if not math.isclose(sum(individual)/3,finite(record[key]["mean"]),abs_tol=1e-5):
            raise ValueError("Seed summary mismatch")
    return {
        "rmse":finite(record["rmse_K"]["mean"]),
        "recall":finite(record["p95_recall"]["mean"]),
        "precision":finite(record["p95_precision"]["mean"]),
        "persistence_rmse":finite(p["rmse_K"]),
        "persistence_recall":finite(p["p95_recall"])
    }

def compare(awi,bcc,new):
    if (new.get("status")!="v512_frozen_historical_neural_forecasts_new_2040_cohort"
        or new.get("frozen_training") is not True
        or new.get("checkpoint_selection_using_2040") is not False
        or new.get("date_plan")!=[3,7,11,15,19,23,27]
        or new.get("year")!=2040):
        raise ValueError("New 2040 verification gate failed")
    index={}
    for group in new.get("rows",[]):
        key=(group["gcm"],group["ssp"])
        if key in index or key not in {(g,s) for g in SOURCES for s in SCENARIOS}:
            raise ValueError("Duplicate or invalid new scenario row")
        if group.get("n_pairs")!=84 or len(group.get("models",{}))!=7:
            raise ValueError("Incomplete new 2040 output")
        index[key]=group
    if len(index)!=4:
        raise ValueError("Missing new GCM×SSP comparisons")
    rows=[]
    for source,original in (("AWI",awi),("BCC",bcc)):
        for scenario in SCENARIOS:
            old=old_groups(source,original,scenario)
            fresh=index[source,scenario]
            for model in MODELS:
                for weight in WEIGHTS:
                    previous=extract_old(source,old,model,weight)
                    current=extract_new(fresh,model,weight)
                    for m in (previous,current):
                        if m["persistence_rmse"]<=0 or m["rmse"]<0:
                            raise ValueError("Invalid persistence-relative skill")
                        for field in ("recall","precision","persistence_recall"):
                            if not 0<=m[field]<=1:
                                raise ValueError("Invalid probability score")
                    old_skill=100*(1-previous["rmse"]/previous["persistence_rmse"])
                    new_skill=100*(1-current["rmse"]/current["persistence_rmse"])
                    rows.append({
                        "gcm":source,"ssp":scenario,"model":model,
                        "historical_tail_weight":weight,
                        "old_years":"2030_2050_2080","old_pairs_per_scenario":72,
                        "new_year":2040,"new_pairs_per_scenario":84,
                        "old_rmse_K":previous["rmse"],
                        "new_rmse_K":current["rmse"],
                        "old_relative_skill_pct":old_skill,
                        "new_relative_skill_pct":new_skill,
                        "relative_skill_difference_pp":new_skill-old_skill,
                        "old_p95_recall":previous["recall"],
                        "new_p95_recall":current["recall"],
                        "new_recall_gap_vs_persistence_pp":100*(current["recall"]-current["persistence_recall"]),
                        "old_p95_precision":previous["precision"],
                        "new_p95_precision":current["precision"],
                    })
    if len(rows)!=24:
        raise ValueError("Missing frozen neural arms")
    summary=[]
    for source in SOURCES:
        for scenario in SCENARIOS:
            group=[x for x in rows if x["gcm"]==source and x["ssp"]==scenario]
            fno1=next(x for x in group if x["model"]=="fno" and x["historical_tail_weight"]==0)
            fno4=next(x for x in group if x["model"]=="fno" and x["historical_tail_weight"]==3)
            summary.append({
                "gcm":source,"ssp":scenario,
                "fno1_new_relative_rmse_skill_pct":fno1["new_relative_skill_pct"],
                "fno1_skill_positive":fno1["new_relative_skill_pct"]>0,
                "fno4_minus_fno1_new_recall_pp":100*(fno4["new_p95_recall"]-fno1["new_p95_recall"]),
                "fno4_minus_fno1_new_rmse_K":fno4["new_rmse_K"]-fno1["new_rmse_K"],
                "persistence_recall_exceeds_fno4":fno4["new_recall_gap_vs_persistence_pp"]<0,
            })
    return {"status":"v513_frozen_2040_vs_old_sparse_projected_forecast_replication_audit",
            "source_runs":{"AWI_original":45864317174,"BCC_original":45872811683,
                           "new_2040":45894027781},
            "model_selection_performed":False,
            "rows":rows,"four_gcm_scenario_summaries":summary,
            "descriptive_counts":{
                "fno1_skill_positive_2040":sum(r["fno1_skill_positive"] for r in summary),
                "fno4_new_recall_higher_than_fno1":sum(r["fno4_minus_fno1_new_recall_pp"]>0 for r in summary),
                "fno4_new_rmse_worse_than_fno1":sum(r["fno4_minus_fno1_new_rmse_K"]>0 for r in summary),
                "persistence_new_recall_higher_than_fno4":sum(r["persistence_recall_exceeds_fno4"] for r in summary)},
            "limitations":[
                "New 2040 dates and old 2030/2050/2080 dates are DIFFERENT samples; do not compare raw Kelvin RMSE as a causal effect",
                "Native persistence relative-skill contrasts are descriptive across time/GCM/scenario",
                "Neither four GCM-scenario cells nor model seeds establish independent climate ensemble inference",
                "Training loss weights remain frozen; no selection based on SSP performance",
                "Monthly sparse sampling does not quantify continuous heatwave duration or forced warming response",
            ]}

def main():
    p=argparse.ArgumentParser()
    for key in ("awi","bcc","new","json-output","csv-output"):
        p.add_argument("--"+key,required=True)
    args=vars(p.parse_args())
    out=Path(args["json_output"]); csv_out=Path(args["csv_output"])
    if out.exists() or csv_out.exists():
        raise FileExistsError("Cannot overwrite frozen audit")
    report=compare(*(json.loads(Path(args[k]).read_text()) for k in ("awi","bcc","new")))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    with csv_out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=report["rows"][0].keys())
        writer.writeheader();writer.writerows(report["rows"])
    print(json.dumps({"status":report["status"],"counts":report["descriptive_counts"]},indent=2))
if __name__=="__main__":main()
