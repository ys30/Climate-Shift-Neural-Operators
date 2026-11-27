"""E1 v5.25 — immutable three-independent-GCM 2050 summer event-day audit.

Compare original v5.20 CAMS, v5.23 BCC and v5.24 AWI archived outputs only.
No future training, additional climate field requests, statistical significance
or independence inflation from SSPs, spatial pixels or initialization seeds.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path

SOURCES={
 "AWI":("v524_AWI_independent_frozen_contiguous_2050_summer_daily_mean_heat_events","AWI-CM-1-1-MR",45937473743,[26,27]),
 "BCC":("v523_BCC_independent_frozen_contiguous_2050_summer_daily_mean_heat_events","BCC-CSM2-MR",45928444026,[23,23]),
 "CAMS":("v520_CAMS_frozen_contiguous_2050_summer_daily_mean_heat_event_pilot","CAMS-CSM1-0",45914481554,None)
}
SCENARIOS=("ssp245","ssp585")
MODELS=("fno","unet")
WEIGHTS=("0","1","3")
SEEDS=(11,22,33)
METRICS=("rmse_K","mae_K","bias_K","daily_p95_recall","daily_p95_precision","event_day_recall","event_day_precision")
STATUS="v525_three_independent_GCM_2050_contiguous_event_day_audit"

def number(v):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
        raise ValueError("Non-finite score or unsupported value")
    return float(v)

def probability(v):
    if v is None:return None
    if not 0 <= number(v) <= 1:raise ValueError("Invalid event or P95 probability")
    return float(v)

def summary_field(rec,key):
    field=rec.get(key)
    if not isinstance(field,dict) or len(field.get("individual",[]))!=3:
        raise ValueError("Incomplete frozen-seed score: "+key)
    vals=field["individual"]
    if all(v is None for v in vals):
        if field.get("mean") is not None:
            raise ValueError("Undefined seed metric must have null mean")
        return None
    if any(v is None for v in vals):
        raise ValueError("Partial undefined seed metric")
    measured=[number(v) for v in vals]
    mean=number(field.get("mean"))
    if not math.isclose(sum(measured)/3,mean,rel_tol=1e-7,abs_tol=1e-6):
        raise ValueError("Source seed mean mismatch: "+key)
    if key.endswith("recall") or key.endswith("precision"):
        for v in measured:probability(v)
    return mean

def validate_source(g,report):
    status,model,_,shape=SOURCES[g]
    if (report.get("status")!=status or report.get("source_model")!=model
        or report.get("year")!=2050 or report.get("source_count_per_ssp")!=91
        or report.get("ssp_fitted") is not False
        or len(report.get("checkpoint_sha",{}))!=18
        or set(report.get("scenarios",{}))!=set(SCENARIOS)):
        raise ValueError("Missing or altered "+g+" original continuous summer source")
    for s in SCENARIOS:
        sc=report["scenarios"][s]
        meta=sc.get("source",{})
        if (sc.get("n_pairs")!=91
            or meta.get("n_source_days")!=91
            or meta.get("n_bounded_regional_native_days_read")!=92):
            raise ValueError("Changed native chronological 91-day source")
        if shape is not None and meta.get("native_grid_shape")!=shape:
            raise ValueError("Changed original GCM native region")
        if g=="AWI" and meta.get("native_calendar")!="proleptic_gregorian":
            raise ValueError("Changed AWI Gregorian calendar")
        if g=="BCC" and meta.get("native_calendar") not in ("noleap","365_day"):
            raise ValueError("Changed BCC noleap calendar")
        if set(sc.get("models",{}))!=set(MODELS):
            raise ValueError("Missing frozen model family")
    return report

def compare(awi,bcc,cams):
    reports={"AWI":awi,"BCC":bcc,"CAMS":cams}
    rows=[]
    for g in SOURCES:
        r=validate_source(g,reports[g])
        for s in SCENARIOS:
            sc=r["scenarios"][s]
            p=sc["persistence"]
            persistence={key:(probability(p[key]) if key.endswith(("recall","precision")) else
                (None if p[key] is None else number(p[key]))) for key in METRICS}
            if persistence["rmse_K"] is None or persistence["rmse_K"]<=0:
                raise ValueError("Invalid Persistence RMSE")
            for model in MODELS:
                if set(sc["models"][model])!=set(WEIGHTS):
                    raise ValueError("Missing immutable training hot loss arm")
                for weight in WEIGHTS:
                    rec=sc["models"][model][weight]
                    model_scores={key:summary_field(rec,key) for key in METRICS}
                    if model_scores["rmse_K"] is None or model_scores["rmse_K"]<0:
                        raise ValueError("Missing frozen RMSE")
                    rows.append({
                      "gcm":g,"scenario":s,"architecture":model,
                      "historical_tail_weight":int(weight),"training_loss_multiplier":1+int(weight),
                      "native_contiguous_target_days":91,
                      "persistence_rmse_K":persistence["rmse_K"],
                      "model_rmse_K":model_scores["rmse_K"],
                      "rmse_skill_pct":100*(1-model_scores["rmse_K"]/persistence["rmse_K"]),
                      "persistence_event_day_recall":persistence["event_day_recall"],
                      "model_event_day_recall":model_scores["event_day_recall"],
                      "persistence_event_day_precision":persistence["event_day_precision"],
                      "model_event_day_precision":model_scores["event_day_precision"],
                      "persistence_daily_p95_recall":persistence["daily_p95_recall"],
                      "model_daily_p95_recall":model_scores["daily_p95_recall"],
                    })
    if len(rows)!=36 or len({(x["gcm"],x["scenario"],x["architecture"],x["historical_tail_weight"]) for x in rows})!=36:
        raise ValueError("Missing or duplicated original neural arms")
    cases=[]
    for g in SOURCES:
        for s in SCENARIOS:
            f0=next(r for r in rows if r["gcm"]==g and r["scenario"]==s
                and r["architecture"]=="fno" and r["historical_tail_weight"]==0)
            f3=next(r for r in rows if r["gcm"]==g and r["scenario"]==s
                and r["architecture"]=="fno" and r["historical_tail_weight"]==3)
            a=f0["model_event_day_recall"];b=f3["model_event_day_recall"]
            p=f0["persistence_event_day_recall"]
            cases.append({
              "gcm":g,"scenario":s,
              "persistence_rmse_K":f0["persistence_rmse_K"],
              "standard_fno_rmse_K":f0["model_rmse_K"],
              "standard_fno_relative_rmse_skill_pct":f0["rmse_skill_pct"],
              "persistence_event_day_recall":p,
              "standard_fno_event_day_recall":a,
              "strong_tail_fno_event_day_recall":b,
              "persistence_minus_standard_fno_event_recall_pp":None if p is None or a is None else 100*(p-a),
              "strong_tail_minus_standard_event_recall_pp":None if a is None or b is None else 100*(b-a),
              "strong_tail_minus_standard_rmse_K":f3["model_rmse_K"]-f0["model_rmse_K"],
              "persistence_minus_strong_tail_event_recall_pp":None if p is None or b is None else 100*(p-b),
              "standard_fno_event_day_precision":f0["model_event_day_precision"],
              "strong_tail_fno_event_day_precision":f3["model_event_day_precision"],
              "persistence_event_day_precision":f0["persistence_event_day_precision"]})
    grouped=[]
    for g in SOURCES:
        x=[r for r in cases if r["gcm"]==g]
        grouped.append({
          "gcm":g,
          "both_ssp_persistence_event_recall_exceeds_standard_fno":all(
             t["persistence_minus_standard_fno_event_recall_pp"] is not None and
             t["persistence_minus_standard_fno_event_recall_pp"]>0 for t in x),
          "both_ssp_strong_tail_improves_event_recall":all(
             t["strong_tail_minus_standard_event_recall_pp"] is not None and
             t["strong_tail_minus_standard_event_recall_pp"]>0 for t in x),
          "both_ssp_standard_fno_improves_rmse":all(
             t["standard_fno_relative_rmse_skill_pct"]>0 for t in x),
          "both_ssp_strong_tail_increases_rmse":all(
             t["strong_tail_minus_standard_rmse_K"]>0 for t in x),
        })
    return {"status":STATUS,
      "source_runs":{g:SOURCES[g][2] for g in SOURCES},
      "independent_GCM_count":3,"scenario_cells":6,
      "neural_arm_rows":36,"frozen_checkpoint_count":54,
      "target_days_per_GCM_scenario":91,
      "ssp_used_for_fitting_or_selection":False,
      "neural_arms":rows,"scenario_findings":cases,"GCM_level_findings":grouped,
      "GCM_level_counts":{
        "persistence_event_recall_exceeds_standard_FNO_both_SSP":sum(t["both_ssp_persistence_event_recall_exceeds_standard_fno"] for t in grouped),
        "strong_tail_increases_event_recall_both_SSP":sum(t["both_ssp_strong_tail_improves_event_recall"] for t in grouped),
        "standard_FNO_RMSE_better_than_persistence_both_SSP":sum(t["both_ssp_standard_fno_improves_rmse"] for t in grouped),
        "strong_tail_increases_RMSE_both_SSP":sum(t["both_ssp_strong_tail_increases_rmse"] for t in grouped)},
      "limits":[
        "Three independent GCMs, one member each; two correlated SSP scenarios per GCM",
        "Summer 2050 91 consecutive simulated daily-mean TAS targets, not observed tasmax heatwaves",
        "Three-day event means same-pixel >=3 adjacent historical training monthly local P95 exceedance days",
        "Recall is same pixel-day overlap, not heatwave object tracking or duration distribution",
        "Winter, long-term trends and CMIP6 forced climate response are not measured",
        "This continuous follow-up was conceived after previous sparse-SSP results, so exploratory not preregistered blind",
        "No temporal/space/seed independence for confidence intervals or statistical significance",
        "Native-grid and P95 differ by GCM; Kelvin comparisons use each GCM's own Persistence baseline"
      ]}

def export(report,out):
    out=Path(out)
    if out.exists():raise FileExistsError("Refuse overwrite of source-verified GCM findings")
    out.mkdir(parents=True)
    (out/"v525_three_GCM_2050_contiguous_event_audit.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for key,name in (("neural_arms","v525_all_36_frozen_arms.csv"),
                     ("scenario_findings","v525_six_GCM_SSP_event_findings.csv"),
                     ("GCM_level_findings","v525_three_GCM_level_findings.csv")):
        rows=report[key]
        with (out/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    lines=["# E1 v5.25 — 2050 continuous summer day-mean temperature event Results",
      "","**Evidence:** original native grids, Historical-only 54 total frozen neural checkpoints; 3 independent GCMs × 2 SSPs × 91 contiguous +1-day targets. This is an exploratory simulated event-day diagnostic, not observed heatwave risk.","",
      "| GCM | SSP | Persistence 3-day event Recall (%) | FNO 1× Recall (%) | FNO 4× Recall (%) | FNO 1× RMSE skill (%) | FNO 4×−1× RMSE (K) |",
      "|---|---|---:|---:|---:|---:|---:|"]
    fmt=lambda x:"undefined" if x is None else f"{100*x:.2f}"
    for r in report["scenario_findings"]:
        lines.append(f"| {r['gcm']} | {r['scenario']} | {fmt(r['persistence_event_day_recall'])} | {fmt(r['standard_fno_event_day_recall'])} | {fmt(r['strong_tail_fno_event_day_recall'])} | {r['standard_fno_relative_rmse_skill_pct']:.2f} | {r['strong_tail_minus_standard_rmse_K']:+.3f} |")
    lines+=["","## Interpretation","",
      "Compare separately each native-grid GCM against its own persistence. Standard FNO may lower total RMSE while missing more sustained-hot event days; strong hot-tail training loss may shift this balance. Do not assume that the strong-loss variant increases Recall in every GCM: the source-verified cases, including any counterexamples, are all reported.","",
      "The six SSP cells correspond to only **three independent GCMs**, so no six-replicate statistical significance claims. Grid points, target days, and three neural seeds also do not multiply GCM-independent evidence.","",
      "Source evidence: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45914481554 ; https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45928444026 ; https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45937473743.",""]
    (out/"v525_three_GCM_contiguous_event_results.md").write_text("\n".join(lines),encoding="utf-8")

def main():
    p=argparse.ArgumentParser()
    for k in ("awi","bcc","cams","output-dir"):p.add_argument("--"+k,required=True)
    a=vars(p.parse_args())
    report=compare(*(json.loads(Path(a[k]).read_text()) for k in ("awi","bcc","cams")))
    export(report,a["output_dir"])
    print(json.dumps({"status":report["status"],"counts":report["GCM_level_counts"],
       "scenario_findings":report["scenario_findings"]},indent=2))
if __name__=="__main__":main()
