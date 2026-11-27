"""E1 v5.22: audit CAMS continuous 2050 SSP summer pilot model/event metrics.

Descriptive event-day recall/precision and RMSE using 91 contiguous target
days per SSP; no significance test and no claim of independent model replication.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
STATUS="v520_CAMS_frozen_contiguous_2050_summer_daily_mean_heat_event_pilot"
OUTSTATUS="v522_CAMS_contiguous_2050_summer_event_tradeoff_audit"
MODELS=("fno","unet")
WEIGHTS=("0","1","3")
SSPS=("ssp245","ssp585")
FIELDS=("rmse_K","mae_K","bias_K","daily_p95_recall","daily_p95_precision",
        "event_day_recall","event_day_precision")
def valid_number(x):
    if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x):
        raise ValueError("Missing or nonfinite archived score")
    return float(x)
def in_unit_interval(x):
    if x is not None and not 0<=valid_number(x)<=1:
        raise ValueError("Probability metric outside [0,1]")
    return x
def inspect(report):
    if (report.get("status")!=STATUS or report.get("year")!=2050
        or report.get("source_model")!="CAMS-CSM1-0"
        or report.get("source_count_per_ssp")!=91
        or report.get("ssp_fitted") is not False
        or len(report.get("checkpoint_sha",{}))!=18
        or set(report.get("scenarios",{}))!=set(SSPS)):
        raise ValueError("Expected complete frozen CAMS 2050 continuous pilot source")
    rows=[];summaries=[];findings=[]
    for scenario in SSPS:
        block=report["scenarios"][scenario]
        source=block["source"]
        if (block.get("n_pairs")!=91 or source.get("n_source_days")!=91
            or source.get("n_bounded_regional_native_days_read")!=92
            or set(block["models"])!=set(MODELS)):
            raise ValueError("Incomplete contiguous summer source coverage")
        persistence=block["persistence"]
        for key in FIELDS:
            if key not in persistence:
                raise ValueError("Missing persistence metric: "+key)
            if persistence[key] is not None:valid_number(persistence[key])
        for key in ("daily_p95_recall","daily_p95_precision","event_day_recall","event_day_precision"):
            in_unit_interval(persistence[key])
        for model in MODELS:
            if set(block["models"][model])!=set(WEIGHTS):
                raise ValueError("Incomplete three frozen tail weights")
            for weight in WEIGHTS:
                score=block["models"][model][weight]
                vals={}
                for key in FIELDS:
                    record=score.get(key)
                    if not isinstance(record,dict) or len(record.get("individual",[]))!=3:
                        raise ValueError("Missing three original seed scores")
                    seed_values=record["individual"]
                    if any(x is None for x in seed_values):
                        if not all(x is None for x in seed_values) or record.get("mean") is not None:
                            raise ValueError("Inconsistent undefined rare-event scores")
                        vals[key]=None
                    else:
                        each=[valid_number(x) for x in seed_values]
                        mu=sum(each)/3
                        if not math.isclose(mu,valid_number(record.get("mean")),abs_tol=1e-6):
                            raise ValueError("Seed mean changed from frozen report")
                        vals[key]=mu
                        if key.endswith("recall") or key.endswith("precision"):
                            for x in each:in_unit_interval(x)
                    for seed,v in zip((11,22,33),seed_values):
                        rows.append({"scenario":scenario,"model":model,
                            "tail_weight":int(weight),"seed":seed,"metric":key,
                            "value":v})
                if vals["rmse_K"] is None or vals["rmse_K"]<0:raise ValueError("Invalid RMSE")
                summaries.append({"scenario":scenario,"model":model,"tail_weight":int(weight),
                  "n_pairs":91,"persistence_rmse_K":valid_number(persistence["rmse_K"]),
                  "model_rmse_K":vals["rmse_K"],
                  "relative_rmse_skill_pct":100*(1-vals["rmse_K"]/valid_number(persistence["rmse_K"])),
                  "persistence_event_day_recall":persistence["event_day_recall"],
                  "model_event_day_recall":vals["event_day_recall"],
                  "persistence_event_day_precision":persistence["event_day_precision"],
                  "model_event_day_precision":vals["event_day_precision"],
                  "model_daily_p95_recall":vals["daily_p95_recall"]})
        fno0=next(x for x in summaries if x["scenario"]==scenario and x["model"]=="fno" and x["tail_weight"]==0)
        fno3=next(x for x in summaries if x["scenario"]==scenario and x["model"]=="fno" and x["tail_weight"]==3)
        if fno0["model_event_day_recall"] is None or fno3["model_event_day_recall"] is None:
            delta=None
        else:
            delta=100*(fno3["model_event_day_recall"]-fno0["model_event_day_recall"])
        gap=None if (persistence["event_day_recall"] is None or fno0["model_event_day_recall"] is None) else 100*(persistence["event_day_recall"]-fno0["model_event_day_recall"])
        findings.append({"scenario":scenario,"n_pairs":91,
           "persistence_event_day_recall":persistence["event_day_recall"],
           "fno_standard_event_day_recall":fno0["model_event_day_recall"],
           "persistence_minus_fno_standard_event_recall_pp":gap,
           "fno_4x_minus_1x_event_recall_pp":delta,
           "fno_4x_minus_1x_rmse_K":fno3["model_rmse_K"]-fno0["model_rmse_K"],
           "fno_standard_rmse_skill_pct":fno0["relative_rmse_skill_pct"],
           "persistence_event_day_precision":persistence["event_day_precision"],
           "fno_standard_event_day_precision":fno0["model_event_day_precision"]})
    if len(summaries)!=12 or len(rows)!=2*2*3*3*len(FIELDS) or len(findings)!=2:
        raise ValueError("Missing CAMS SSP arm or frozen seed")
    return {"status":OUTSTATUS,
      "source_run":45914481554,"source_model":"CAMS-CSM1-0",
      "n_independent_GCMs":1,"n_SSP_scenarios":2,
      "n_contiguous_prediction_days_per_scenario":91,
      "future_training_performed":False,
      "scenario_findings":findings,"arm_summaries":summaries,"seed_metric_rows":rows,
      "limits":["Exploratory CAMS-only one-member 2050 summer evaluation; SSP scenarios are correlated",
        "Event-day recall requires pixel-day overlap of a >=3-day hot spell; no object matching",
        "Native simulated tas daily MEAN above CAMS training-only local monthly P95, not observed tasmax",
        "First/last event windows censored at summer boundaries",
        "No independence assumption across days, pixels or training seeds",
        "No future-SSP metric-based checkpoint selection or inferential p-value"]}

def export(result,out):
    out=Path(out)
    if out.exists():raise FileExistsError("Do not overwrite v5.22 result")
    out.mkdir(parents=True)
    (out/"v522_contiguous_CAMS_event_tradeoff.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for key,name in (("scenario_findings","v522_scenario_findings.csv"),
                     ("arm_summaries","v522_model_arm_summary.csv"),
                     ("seed_metric_rows","v522_original_seed_metrics.csv")):
        rows=result[key]
        with (out/name).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    lines=["# E1 v5.22 — CAMS 2050 contiguous summer: source-verified exploratory results",
      "",
      "A GCM-simulated daily-mean TAS hot event requires at least three consecutive target days above the historical-training local monthly P95 at the same native grid cell. The evaluation covers 91 consecutive target days per scenario for one CAMS member. It does not measure observed heatwaves or establish cross-GCM replication.","",
      "| Scenario | Persistence event-day Recall | Standard FNO event-day Recall | Gap (pp; Persistence minus FNO) | FNO 4×−1× event Recall (pp) | FNO 4×−1× RMSE (K) |",
      "|---|---:|---:|---:|---:|---:|"]
    fmt=lambda x: "undefined" if x is None else f"{x:.3f}"
    for r in result["scenario_findings"]:
        lines.append(f"| {r['scenario']} | {fmt(r['persistence_event_day_recall'])} | {fmt(r['fno_standard_event_day_recall'])} | {fmt(r['persistence_minus_fno_standard_event_recall_pp'])} | {fmt(r['fno_4x_minus_1x_event_recall_pp'])} | {fmt(r['fno_4x_minus_1x_rmse_K'])} |")
    lines+=["","## Interpretation","",
        "A within-CAMS, two-scenario descriptive comparison only. Event-day Recall must not be conflated with one-day exceedance Recall. For fair interpretation, report Precision and weighted RMSE alongside Recall, because threshold-heavy models can increase hits by over-predicting events. This pilot did not pre-register a multi-model long-duration generalization test.","",
        "Source: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45914481554",""]
    (out/"v522_CAMS_contiguous_event_results.md").write_text("\n".join(lines),encoding="utf-8")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output-dir",required=True)
    a=p.parse_args()
    result=inspect(json.loads(Path(a.input).read_text()))
    export(result,a.output_dir)
    print(json.dumps({"status":result["status"],"findings":result["scenario_findings"]},indent=2))
if __name__=="__main__":main()
