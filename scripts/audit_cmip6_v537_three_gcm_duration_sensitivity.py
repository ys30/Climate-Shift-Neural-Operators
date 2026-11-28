"""E1 v5.37: source-locked three-GCM 2/3/5-day duration sensitivity audit.

Use only successfully archived AWI v5.34, BCC v5.35, CAMS v5.36.
Cross-GCM comparisons are descriptive, never infer 36 independent GCMs.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
SOURCES={
 "AWI":("v534_AWI_2040_2050_two_three_five_day_original_frozen_sensitivity",46109631398),
 "BCC":("v535_BCC_2040_2050_original_frozen_two_three_five_day_event_sensitivity",46117110466),
 "CAMS":("v536_CAMS_2040_2050_original_frozen_2_3_5_day_event_sensitivity",46117486346)}
STATUS="v537_three_GCM_two_year_two_three_five_day_sensitivity_audit"
YEARS=(2040,2050)
SSPS=("ssp245","ssp585")
DURATIONS=(2,3,5)
CHOICES=(("persistence",None),("fno",1),("fno",4))

def score(x):
    if x is None:return None
    if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or not 0<=x<=1:
        raise ValueError("Invalid source event score")
    return float(x)

def verified_source(g,source):
    if (source.get("status")!=SOURCES[g][0]
        or source.get("years")!=list(YEARS)
        or source.get("scenarios")!=list(SSPS)
        or source.get("durations")!=list(DURATIONS)
        or source.get("ssp_fitted") is not False
        or source.get("all_original_three_day_reference_scores_reproduced") is not True
        or source.get("n_original_frozen_checkpoints")!=18
        or source.get("independent_GCM_count")!=1
        or len(source.get("checkpoint_sha",{}))!=18):
        raise ValueError("Original frozen GCM source incomplete or changed")
    rows=source.get("records",[])
    expected={(y,s,d,m,w) for y in YEARS for s in SSPS for d in DURATIONS for m,w in CHOICES}
    if len(rows)!=36 or {
        (r["year"],r["scenario"],r["min_days"],r["model"],r["historical_multiplier"]) for r in rows
    }!=expected:raise ValueError("Missing/duplicate native duration record")
    for r in rows:
        for k in ("event_day_recall","event_day_precision"):
            score(r[k])
        for k in ("observed_event_day_area_weight","predicted_event_day_area_weight"):
            v=r.get(k)
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<0:
                raise ValueError("Invalid original area-weighted event support")
        if (r["observed_event_day_area_weight"]==0)!=(r["event_day_recall"] is None):
            raise ValueError("Undefined Recall must have zero observed event days")
        if (r["predicted_event_day_area_weight"]==0)!=(r["event_day_precision"] is None):
            raise ValueError("Undefined Precision must have zero forecast event days")
    # A longer run definition cannot add days to a fixed observed event cohort.
    # Native climate grids and GCM-specific area weights remain separate.
    for y in YEARS:
        for s in SSPS:
            for d in DURATIONS:
                subset=[r for r in rows if (r["year"],r["scenario"],r["min_days"])==(y,s,d)]
                obs=[r["observed_event_day_area_weight"] for r in subset]
                if max(obs)-min(obs)>1e-6*max(1.,max(obs)):
                    raise ValueError("Unequal true-event denominator across alternatives")
            base={r["min_days"]:r["observed_event_day_area_weight"] for r in rows
                  if r["year"]==y and r["scenario"]==s and r["model"]=="persistence"}
            if not (base[2]+1e-6>=base[3]>=base[5]-1e-6):
                raise ValueError("Native observed run-day membership must shrink with duration")
    return rows

def audit(sources):
    if set(sources)!=set(SOURCES):raise ValueError("Exactly three original GCM archives required")
    all_rows=[]
    for g in SOURCES:
        for r in verified_source(g,sources[g]):
            all_rows.append({"gcm":g,**r})
    if len(all_rows)!=108:raise ValueError("Three GCM sources require exactly 108 rows")
    cells=[]
    for g in SOURCES:
        for y in YEARS:
            for s in SSPS:
                for d in DURATIONS:
                    z={(r["model"],r["historical_multiplier"]):r for r in all_rows
                       if (r["gcm"],r["year"],r["scenario"],r["min_days"])==(g,y,s,d)}
                    p=z["persistence",None];f1=z["fno",1];f4=z["fno",4]
                    gap=None if p["event_day_recall"] is None or f1["event_day_recall"] is None else 100*(p["event_day_recall"]-f1["event_day_recall"])
                    gain=None if f1["event_day_recall"] is None or f4["event_day_recall"] is None else 100*(f4["event_day_recall"]-f1["event_day_recall"])
                    loss=None if f1["event_day_precision"] is None or f4["event_day_precision"] is None else 100*(f4["event_day_precision"]-f1["event_day_precision"])
                    cells.append({
                      "gcm":g,"year":y,"scenario":s,"min_days":d,
                      "observed_event_day_area_weight_native":p["observed_event_day_area_weight"],
                      "persistence_event_recall":p["event_day_recall"],
                      "fno_1x_event_recall":f1["event_day_recall"],
                      "fno_4x_event_recall":f4["event_day_recall"],
                      "persistence_minus_fno_1x_recall_pp":gap,
                      "fno_4x_minus_1x_recall_pp":gain,
                      "fno_4x_minus_1x_precision_pp":loss})
    changes=[]
    for g in SOURCES:
        for y in YEARS:
            for s in SSPS:
                r2=next(r for r in cells if (r["gcm"],r["year"],r["scenario"],r["min_days"])==(g,y,s,2))
                r5=next(r for r in cells if (r["gcm"],r["year"],r["scenario"],r["min_days"])==(g,y,s,5))
                key="persistence_minus_fno_1x_recall_pp"
                changes.append({"gcm":g,"year":y,"scenario":s,
                    "recall_gap_2day_pp":r2[key],
                    "recall_gap_5day_pp":r5[key],
                    "five_minus_two_day_gap_change_pp":None if r2[key] is None or r5[key] is None else r5[key]-r2[key],
                    "fno_4x_gain_2day_pp":r2["fno_4x_minus_1x_recall_pp"],
                    "fno_4x_gain_5day_pp":r5["fno_4x_minus_1x_recall_pp"],
                    "five_minus_two_day_hotweight_gain_change_pp":None if r2["fno_4x_minus_1x_recall_pp"] is None or r5["fno_4x_minus_1x_recall_pp"] is None else r5["fno_4x_minus_1x_recall_pp"]-r2["fno_4x_minus_1x_recall_pp"]})
    counts={}
    for d in DURATIONS:
        sub=[r for r in cells if r["min_days"]==d]
        counts[str(d)]={"valid_scenarios":len(sub),
          "persistence_recall_above_fno_1x":sum(r["persistence_minus_fno_1x_recall_pp"] is not None and r["persistence_minus_fno_1x_recall_pp"]>0 for r in sub),
          "fno_4x_recall_above_fno_1x":sum(r["fno_4x_minus_1x_recall_pp"] is not None and r["fno_4x_minus_1x_recall_pp"]>0 for r in sub),
          "fno_4x_precision_below_fno_1x":sum(r["fno_4x_minus_1x_precision_pp"] is not None and r["fno_4x_minus_1x_precision_pp"]<0 for r in sub),
          "undefined_recall_gap":sum(r["persistence_minus_fno_1x_recall_pp"] is None for r in sub)}
    return {"status":STATUS,"independent_GCM_count":3,"GCM_year_SSP_cells":12,
       "GCM_year_SSP_duration_cells":36,"original_frozen_checkpoints":54,
       "original_GCM_duration_arm_rows":108,"no_SSP_fitting":True,
       "all_three_day_archived_references_reproduced":True,
       "source_runs":{g:SOURCES[g][1] for g in SOURCES},
       "duration_counts":counts,"all_108_source_rows":all_rows,
       "scenario_duration_findings":cells,"paired_two_vs_five_day_changes":changes,
       "limitations":[
         "Exactly 3 independent GCM members; 12 GCM-year-SSP cells and 36 duration cases are correlated within GCM",
         "Event duration definitions 2/5 days were analyzed after original 3-day results: exploratory, not preregistered",
         "Fixed Historical monthly P95, original 54 frozen checkpoints; future SSP never used for training",
         "91-day native grid simulated daily-mean TAS; censored at beginning/end of evaluated summers",
         "The duration thresholds label qualifying same-pixel EVENT-DAYS, not tracked heatwave lengths",
         "Undefined Recall or Precision when no observed or predicted qualifying event days; no imputation",
         "No spatial, temporal, SSP, or seed pseudo-replicate p-values, no 36-independent-GCM claim"]}

def write(output,path):
    path=Path(path)
    if path.exists():raise FileExistsError("Do not overwrite original three-GCM duration audit")
    path.mkdir(parents=True)
    (path/"v537_three_GCM_duration_audit.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    for key,filename in (("scenario_duration_findings","v537_36_duration_cases.csv"),
                         ("paired_two_vs_five_day_changes","v537_12_duration_changes.csv"),
                         ("all_108_source_rows","v537_108_original_model_duration_rows.csv")):
        with (path/filename).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(output[key][0]))
            w.writeheader();w.writerows(output[key])
    lines=["# E1 v5.37 — Three-GCM 2/3/5-day event-definition sensitivity","",
       "All values originate from three previously successful GCM-specific duration archives. 54 original Historical-only checkpoints and monthly local P95 were held fixed. The original three-day scores were required to reproduce their immutable 2040/2050 counterparts.","",
       "| Minimum days | GCM–year–SSP cells | Persistence Recall > FNO 1× | FNO 4× Recall > 1× | FNO 4× Precision < 1× | Undefined Recall gap |",
       "|---:|---:|---:|---:|---:|---:|"]
    for duration,record in output["duration_counts"].items():
        lines.append(f"| {duration} | {record['valid_scenarios']} | {record['persistence_recall_above_fno_1x']} | {record['fno_4x_recall_above_fno_1x']} | {record['fno_4x_precision_below_fno_1x']} | {record['undefined_recall_gap']} |")
    lines+=["","## Interpretation","",
       "These are *descriptive sensitivity* results rather than independent tests. Compare the full 12 paired two-versus-five-day differences rather than assuming Recall gaps must monotonically increase with duration: different durations change the denominator and event cohort, and some models may have undefined event rates. Report every reversal and no-event case.","",
       "The three GCM members are the only independent climate-model units. Two years, two SSP scenarios, and three duration definitions all involve correlated measurements within GCM. The metric concerns coincident event-days of native daily mean TAS above own Historical month-local P95, not objects/tracked heatwaves, tasmax extremes or observed exposure. Only 91 summer targets per case, with boundary censoring. No p-values or false independence claims.","",
       "Source archive runs: "+", ".join(f"{g}: {SOURCES[g][1]}" for g in SOURCES),""]
    (path/"E1_v537_duration_sensitivity_manuscript_results.md").write_text("\n".join(lines),encoding="utf-8")

def main():
    p=argparse.ArgumentParser()
    for g in SOURCES:p.add_argument("--"+g.lower(),required=True)
    p.add_argument("--output-dir",required=True)
    args=p.parse_args()
    d={g:json.loads(Path(getattr(args,g.lower())).read_text()) for g in SOURCES}
    r=audit(d);write(r,args.output_dir)
    print(json.dumps({"status":STATUS,"independent_GCM":3,"duration_cases":36,
                      "duration_counts":r["duration_counts"]},indent=2))
if __name__=="__main__":main()
