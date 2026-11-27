"""E1 v5.32 — source-locked paired 2040 vs 2050 summer replication.

Three independent CMIP6 GCMs; two paired SSPs/model and two separated years.
All 18 original frozen Historical-only checkpoints/model must have identical SHA
across years. No post-SSP fitting or selection; descriptive only.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from scripts.audit_cmip6_v525_three_gcm_contiguous_events import (
    SOURCES as Y2050,summary_field,number,probability,SCENARIOS,MODELS,WEIGHTS)

Y2040={
 "AWI":("v529_AWI_predeclared_2040_independent_year_contiguous_summer",45954950987),
 "BCC":("v530_BCC_independent_2040_contiguous_summer_holdout",45956507929),
 "CAMS":("v531_CAMS_2040_predeclared_contiguous_summer_temporal_holdout",46054222652)}
RUNS_2050={g:info[2] for g,info in Y2050.items()}
STATUS="v532_three_GCM_two_summer_years_frozen_event_replication"
KEYS=("rmse_K","event_day_recall","event_day_precision")

def validate(g,year,data,reference_sha=None):
    wanted=(Y2040[g][0] if year==2040 else Y2050[g][0])
    if (data.get("status")!=wanted or data.get("source_model")!=Y2050[g][1]
        or data.get("year")!=year or data.get("source_count_per_ssp")!=91
        or data.get("ssp_fitted") is not False
        or len(data.get("checkpoint_sha",{}))!=18
        or set(data.get("scenarios",{}))!=set(SCENARIOS)):
        raise ValueError(f"{g} {year} original frozen source contract changed")
    if reference_sha is not None and data["checkpoint_sha"]!=reference_sha:
        raise ValueError("Frozen Historical checkpoint hashes differ between 2040 and 2050")
    for scenario in SCENARIOS:
        d=data["scenarios"][scenario]; m=d["source"]
        if (d.get("n_pairs")!=91 or m.get("n_bounded_regional_native_days_read")!=92
            or m.get("n_source_days")!=91
            or set(d.get("models",{}))!=set(MODELS)):
            raise ValueError("Incomplete or changed native 91-day source")
        shape=Y2050[g][3]
        if shape is not None and m.get("native_grid_shape")!=shape:
            raise ValueError("Changed model native grid")
        if g=="AWI" and m.get("native_calendar")!="proleptic_gregorian":
            raise ValueError("AWI Gregorian calendar changed")
        if g=="BCC" and m.get("native_calendar") not in ("noleap","365_day"):
            raise ValueError("BCC noleap calendar changed")
        for model in MODELS:
            if set(d["models"][model])!=set(WEIGHTS):
                raise ValueError("Incomplete original Historical frozen loss arms")
            for w in WEIGHTS:
                for metric in KEYS:
                    summary_field(d["models"][model][w],metric)
        for metric in KEYS:
            x=d["persistence"].get(metric)
            if x is None: raise ValueError("Missing native Persistence score")
            probability(x) if metric!="rmse_K" else number(x)
    return data

def analyze(reports):
    if set(reports)!=set(Y2050):raise ValueError("Require three original GCMs")
    rows=[]
    for g in Y2050:
        data=reports[g]
        if set(data)!={2040,2050}:raise ValueError("Require paired 2040 and 2050")
        ref=validate(g,2050,data[2050])
        verify=validate(g,2040,data[2040],ref["checkpoint_sha"])
        for year in (2040,2050):
            for scenario in SCENARIOS:
                d=data[year]["scenarios"][scenario]
                base=d["persistence"]
                for model in MODELS:
                    for weight in WEIGHTS:
                        rec=d["models"][model][weight]
                        rmse=summary_field(rec,"rmse_K")
                        recall=summary_field(rec,"event_day_recall")
                        precision=summary_field(rec,"event_day_precision")
                        rows.append({
                          "gcm":g,"year":year,"scenario":scenario,"model":model,
                          "historical_tail_weight":int(weight),"training_multiplier":1+int(weight),
                          "days":91,
                          "persistence_rmse_K":number(base["rmse_K"]),
                          "model_rmse_K":rmse,
                          "rmse_skill_pct":100*(1-rmse/number(base["rmse_K"])),
                          "persistence_event_recall":probability(base["event_day_recall"]),
                          "model_event_recall":recall,
                          "persistence_event_precision":probability(base["event_day_precision"]),
                          "model_event_precision":precision})
    if len(rows)!=72 or len({(z["gcm"],z["year"],z["scenario"],z["model"],z["historical_tail_weight"]) for z in rows})!=72:
        raise ValueError("Expected 72 distinct two-year original frozen neural arms")
    cases=[]
    for g in Y2050:
        for year in (2040,2050):
            for scenario in SCENARIOS:
                choices={x["historical_tail_weight"]:x for x in rows
                         if x["gcm"]==g and x["year"]==year and x["scenario"]==scenario and x["model"]=="fno"}
                f0=choices[0];f4=choices[3]
                result={
                  "gcm":g,"year":year,"scenario":scenario,
                  "persistence_event_recall":f0["persistence_event_recall"],
                  "standard_fno_event_recall":f0["model_event_recall"],
                  "strong_tail_fno_event_recall":f4["model_event_recall"],
                  "persistence_minus_fno_recall_pp":100*(f0["persistence_event_recall"]-f0["model_event_recall"]),
                  "four_x_minus_one_x_recall_pp":100*(f4["model_event_recall"]-f0["model_event_recall"]),
                  "four_x_minus_one_x_precision_pp":100*(f4["model_event_precision"]-f0["model_event_precision"]),
                  "standard_fno_rmse_skill_pct":f0["rmse_skill_pct"],
                  "four_x_minus_one_x_rmse_K":f4["model_rmse_K"]-f0["model_rmse_K"],
                }
                cases.append(result)
    assert len(cases)==12
    pairs=[]
    for g in Y2050:
        for scenario in SCENARIOS:
            now=next(r for r in cases if (r["gcm"],r["year"],r["scenario"])==(g,2040,scenario))
            later=next(r for r in cases if (r["gcm"],r["year"],r["scenario"])==(g,2050,scenario))
            pairs.append({"gcm":g,"scenario":scenario,
               "standard_fno_recall_2040":now["standard_fno_event_recall"],
               "standard_fno_recall_2050":later["standard_fno_event_recall"],
               "standard_fno_2050_minus_2040_recall_pp":100*(later["standard_fno_event_recall"]-now["standard_fno_event_recall"]),
               "persistence_recall_gap_2040_pp":now["persistence_minus_fno_recall_pp"],
               "persistence_recall_gap_2050_pp":later["persistence_minus_fno_recall_pp"],
               "four_x_recall_gain_2040_pp":now["four_x_minus_one_x_recall_pp"],
               "four_x_recall_gain_2050_pp":later["four_x_minus_one_x_recall_pp"],
               "standard_fno_rmse_skill_2040_pct":now["standard_fno_rmse_skill_pct"],
               "standard_fno_rmse_skill_2050_pct":later["standard_fno_rmse_skill_pct"]})
    predicates={
      "persistence_event_recall_exceeds_standard_FNO":lambda r:r["persistence_minus_fno_recall_pp"]>0,
      "four_x_loss_increases_event_recall":lambda r:r["four_x_minus_one_x_recall_pp"]>0,
      "standard_FNO_reduces_RMSE":lambda r:r["standard_fno_rmse_skill_pct"]>0,
      "four_x_loss_reduces_event_precision":lambda r:r["four_x_minus_one_x_precision_pp"]<0,
      "four_x_loss_increases_RMSE":lambda r:r["four_x_minus_one_x_rmse_K"]>0,
    }
    counts={key:sum(test(r) for r in cases) for key,test in predicates.items()}
    gcm_counts={key:sum(all(test(r) for r in cases if r["gcm"]==g) for g in Y2050)
                for key,test in predicates.items()}
    return {"status":STATUS,"independent_GCM_count":3,"distinct_summer_years":[2040,2050],
      "paired_SSP_per_GCM":list(SCENARIOS),"year_GCM_SSP_cells":12,
      "original_frozen_neural_checkpoint_count":54,"descriptive_neural_arm_rows":72,
      "source_runs":{g:{"2040":Y2040[g][1],"2050":RUNS_2050[g]} for g in Y2050},
      "checkpoint_SHA_equal_between_years":True,"ssp_used_for_fitting":False,
      "complete_neural_arms":rows,"scenario_findings":cases,"paired_year_findings":pairs,
      "cell_counts":counts,"GCM_level_consistency_counts":gcm_counts,
      "limitations":[
       "Exactly three GCM members, two correlated SSPs each and two correlated years per GCM",
       "Distinct evaluation years do not create six independent climate-model members",
       "2040 selected as a follow-up after exploratory 2050 results; not independent design preregistration",
       "Historical-only frozen weights and P95 thresholds verified matching between 2040 and 2050",
       "Future simulated daily-mean native tas, not observed tasmax, human heat exposure or tracked heatwave objects",
       "Same-pixel three-consecutive-day event-day masks are censored at summer boundaries",
       "No p-values or uncertainty inference from spatial cells, days, SSPs, or three random seeds",
       "Post-hoc comparison does not justify selecting a winning model for new future data"]}

def write(r,folder):
    folder=Path(folder)
    if folder.exists():raise FileExistsError("Do not overwrite source-verified two-year result")
    folder.mkdir(parents=True)
    (folder/"v532_three_GCM_2040_2050_event_replication.json").write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    for key,name in (("complete_neural_arms","v532_all_72_model_scenario_year_rows.csv"),
                     ("scenario_findings","v532_12_scenario_year_findings.csv"),
                     ("paired_year_findings","v532_six_2040_2050_pairs.csv")):
        with (folder/name).open("w",newline="",encoding="utf-8") as f:
            wr=csv.DictWriter(f,fieldnames=list(r[key][0]))
            wr.writeheader();wr.writerows(r[key])
    lines=["# E1 v5.32 — independent-year replication of sustained high-TAS event diagnostics","",
      "Original Historical-only checkpoints were unchanged. 3 independently simulated native-grid GCM members × 2 separate years (2040, 2050) × 2 paired SSP pathways; 91 consecutive future simulated summer target days in each cell. The six pairs represent only three independent GCM members.","",
      "| GCM | Year | SSP | Persistence event Recall % | FNO 1× event Recall % | FNO 4× event Recall % | FNO 1× RMSE skill % | FNO 4×−1× event Precision pp |",
      "|---|---:|---|---:|---:|---:|---:|---:|"]
    for z in r["scenario_findings"]:
        lines.append(f"| {z['gcm']} | {z['year']} | {z['scenario']} | {100*z['persistence_event_recall']:.2f} | {100*z['standard_fno_event_recall']:.2f} | {100*z['strong_tail_fno_event_recall']:.2f} | {z['standard_fno_rmse_skill_pct']:.2f} | {z['four_x_minus_one_x_precision_pp']:+.2f} |")
    lines+=["","## Descriptive repeatability",""]
    for key,v in r["cell_counts"].items():
        lines.append(f"- {key}: {v}/12 paired model–year–SSP cells; GCM consistency: {r['GCM_level_consistency_counts'][key]}/3.")
    lines+=["","## Limits and next step","",
      "These are descriptive, not inferential tests; 2040 is a follow-up year selected after the exploratory 2050 study. The relevant generalization unit is the GCM member (n=3), and SSP/year samples are correlated within GCM. Simulated daily mean TAS P95 event-day overlap is not observed heatwave exposure or event-object skill. Future replication should use additional independently simulated GCM members and years, explicit event duration diagnostics and predeclared calibration thresholds. Do not pool all 12 as independent climate-model replications.",""]
    (folder/"v532_two_year_three_GCM_results.md").write_text("\n".join(lines),encoding="utf-8")
def main():
    ap=argparse.ArgumentParser()
    for g in Y2050:
        for year in (2040,2050):
            ap.add_argument(f"--{g.lower()}-{year}",required=True)
    ap.add_argument("--output-dir",required=True)
    a=vars(ap.parse_args())
    d={g:{year:json.loads(Path(a[f"{g.lower()}_{year}"]).read_text()) for year in (2040,2050)} for g in Y2050}
    out=analyze(d);write(out,a["output_dir"])
    print(json.dumps({"status":out["status"],"cells":out["year_GCM_SSP_cells"],
       "GCM_count":out["independent_GCM_count"],"counts":out["cell_counts"],
       "GCM_consistency":out["GCM_level_consistency_counts"]},indent=2))
if __name__=="__main__":main()
