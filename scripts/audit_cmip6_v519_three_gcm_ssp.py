"""E1 v5.19: auditable three-independent-GCM historical-frozen SSP forecast comparison.

Only model-specific, same-cohort Persistence-relative skills are combined.
The six GCM×SSP cases are not six independent climate models.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from scripts.compare_cmip6_v513_2040_replication import old_groups,extract_old,finite

GCM=("AWI","BCC","CAMS")
SCENARIOS=("ssp245","ssp585")
MODELS=("fno","unet")
WEIGHTS=(0,1,3)
CAMS_STATUS="v518_CAMS_Historical_frozen_sparse_SSP_predictive_skill"
STATUS="v519_three_independent_GCM_frozen_sparse_SSP_audit"

def cams_group(report,scenario):
    if (report.get("status")!=CAMS_STATUS
        or report.get("source_model")!="CAMS-CSM1-0"
        or report.get("ssp_used_for_fitting_or_checkpoint_selection") is not False
        or report.get("future_years")!=[2030,2050,2080]
        or report.get("future_month_days")!=[8,20]
        or len(report.get("checkpoint_sha",{}))!=18):
        raise ValueError("Unverified CAMS historical-frozen SSP source")
    if set(report["scenario_results"])!=set(SCENARIOS):
        raise ValueError("Incomplete CAMS scenario coverage")
    return report["scenario_results"][scenario]["scores"]["all"]

def extract_cams(group,model,weight):
    if group.get("n_pairs")!=72:
        raise ValueError("Expected 72 fixed CAMS source-target pairs")
    rec=group["models"][model][str(weight)]
    base=group["persistence"]
    return {
       "rmse":finite(rec["weighted_rmse_K"]["mean"]),
       "recall":finite(rec["local_p95_recall"]["mean"]),
       "precision":finite(rec["local_p95_precision"]["mean"]),
       "persistence_rmse":finite(base["weighted"]["rmse"]),
       "persistence_recall":finite(base["monthly_hot"]["hot_recall"]),
    }

def compare(awi,bcc,cams):
    sources={"AWI":awi,"BCC":bcc,"CAMS":cams}
    rows=[]
    for g in GCM:
        for scenario in SCENARIOS:
            group=cams_group(cams,scenario) if g=="CAMS" else old_groups(g,sources[g],scenario)
            for m in MODELS:
                for w in WEIGHTS:
                    r=extract_cams(group,m,w) if g=="CAMS" else extract_old(g,group,m,w)
                    if r["persistence_rmse"]<=0 or r["rmse"]<0:
                        raise ValueError("Invalid model or Persistence RMSE")
                    for key in ("recall","precision","persistence_recall"):
                        if not 0<=r[key]<=1:
                            raise ValueError("Invalid historical-P95 extreme metric")
                    rows.append({
                      "gcm":g,"ssp":scenario,"model":m,"historical_tail_weight":w,
                      "n_pairs":72,
                      "persistence_rmse_K":r["persistence_rmse"],
                      "model_rmse_K":r["rmse"],
                      "relative_rmse_skill_pct":100*(1-r["rmse"]/r["persistence_rmse"]),
                      "persistence_local_p95_recall":r["persistence_recall"],
                      "model_local_p95_recall":r["recall"],
                      "model_local_p95_precision":r["precision"],
                      "model_recall_minus_persistence_pp":100*(r["recall"]-r["persistence_recall"]),
                    })
    if len(rows)!=36:raise ValueError("Missing one or more GCM SSP neural training arms")
    findings=[]
    for g in GCM:
        for s in SCENARIOS:
            rr=[r for r in rows if r["gcm"]==g and r["ssp"]==s]
            fno0=next(r for r in rr if r["model"]=="fno" and r["historical_tail_weight"]==0)
            fno3=next(r for r in rr if r["model"]=="fno" and r["historical_tail_weight"]==3)
            findings.append({
              "gcm":g,"ssp":s,"persistence_rmse_K":fno0["persistence_rmse_K"],
              "fno1_rmse_K":fno0["model_rmse_K"],
              "fno1_rmse_skill_pct":fno0["relative_rmse_skill_pct"],
              "fno1_beats_persistence_rmse":fno0["relative_rmse_skill_pct"]>0,
              "fno4_recall_minus_fno1_pp":100*(fno3["model_local_p95_recall"]-fno0["model_local_p95_recall"]),
              "fno4_rmse_minus_fno1_K":fno3["model_rmse_K"]-fno0["model_rmse_K"],
              "persistence_recall_minus_fno4_pp":100*(fno3["persistence_local_p95_recall"]-fno3["model_local_p95_recall"]),
            })
    grouped=[]
    for g in GCM:
        entries=[x for x in findings if x["gcm"]==g]
        grouped.append({
            "gcm":g,
            "ssp245_fno1_relative_skill_pct":next(x["fno1_rmse_skill_pct"] for x in entries if x["ssp"]=="ssp245"),
            "ssp585_fno1_relative_skill_pct":next(x["fno1_rmse_skill_pct"] for x in entries if x["ssp"]=="ssp585"),
            "both_ssp_fno1_improve_rmse":all(x["fno1_beats_persistence_rmse"] for x in entries),
            "both_ssp_fno4_improve_recall_over_fno1":all(x["fno4_recall_minus_fno1_pp"]>0 for x in entries),
            "both_ssp_fno4_worsen_rmse_vs_fno1":all(x["fno4_rmse_minus_fno1_K"]>0 for x in entries),
            "both_ssp_persistence_recall_exceeds_fno4":all(x["persistence_recall_minus_fno4_pp"]>0 for x in entries),
        })
    counts={
       "gcm_with_both_ssp_fno1_rmse_advantage":sum(x["both_ssp_fno1_improve_rmse"] for x in grouped),
       "gcm_with_both_ssp_fno4_recall_advantage":sum(x["both_ssp_fno4_improve_recall_over_fno1"] for x in grouped),
       "gcm_with_both_ssp_fno4_rmse_tradeoff":sum(x["both_ssp_fno4_worsen_rmse_vs_fno1"] for x in grouped),
       "gcm_with_both_ssp_persistence_higher_recall":sum(x["both_ssp_persistence_recall_exceeds_fno4"] for x in grouped),
    }
    return {"status":STATUS,
        "source_runs":{"AWI":45864317174,"BCC":45872811683,"CAMS":45905955958},
        "training_Historical_only":True,
        "ssp_used_for_checkpoint_selection":False,
        "dates_per_scenario_per_GCM":72,
        "independent_GCM_count":3,"scenario_cells":6,
        "arms":rows,"gcm_ssp_findings":findings,
        "gcm_grouped_descriptive_findings":grouped,
        "descriptive_counts":counts,
        "scientific_limits":[
          "Six GCM×SSP cells are clustered within three different GCMs, not six independent models",
          "Each GCM trains independent normalizer, P95 and native grid, so regional Kelvin RMSE should be compared relative to its own Persistence",
          "Predictions target GCM-simulated daily mean TAS (not real future observations)",
          "Sparse 24 pairs/year cannot estimate heatwave duration, trends, annual climatologies, or forced climate-change response",
          "Historical 2005–2014 is checkpoint-selection validation, not pristine independent testing",
          "Within-GCM SSP245 and SSP585 share GCM dynamics and cannot provide independent GCM replicates",
          "No SSP tuning or selecting a winning loss arm using future data",
          "No p-value or causal generalization from three native-grid GCMs"
        ]}

def main():
    p=argparse.ArgumentParser()
    for k in ("awi","bcc","cams","json-output","csv-output"):
        p.add_argument("--"+k,required=True)
    a=vars(p.parse_args())
    output=Path(a["json_output"]);csvout=Path(a["csv_output"])
    if output.exists() or csvout.exists():raise FileExistsError("Never overwrite three-GCM evidence")
    d=compare(*(json.loads(Path(a[k]).read_text()) for k in ("awi","bcc","cams")))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")
    with csvout.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=d["arms"][0].keys())
        writer.writeheader();writer.writerows(d["arms"])
    print(json.dumps({"status":d["status"],"descriptive_counts":d["descriptive_counts"],
      "findings":d["gcm_ssp_findings"]},indent=2))
if __name__=="__main__":main()
