"""E1 v5.36 CAMS native 2040/2050 2/3/5-day sustained TAS sensitivity.

Use CAMS Historical-only monthly local TAS P95 and exactly the original 18
SHA-verified frozen checkpoints; no SSP refitting. The original 3-day
event-day Recall/Precision for Persistence and FNO 1x/4x (each frozen
three-seed mean) must reproduce 2040 v5.31 and 2050 v5.20 archives.
This remains one GCM with two correlated SSPs and two simulated years.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.evaluate_cmip6_v518_cams_frozen_ssp import inspect_frozen,manifest_for_CAMS
from scripts.audit_cmip6_catalog import read_catalog,select_pairs
from scripts.evaluate_cmip6_v520_cams_contiguous_summer import real_future_91,fixed_prediction
from scripts.evaluate_cmip6_v534_awi_duration_sensitivity import (
    event,check_match,summarize,YEARS,SSPS,DURATIONS,SEEDS,WEIGHTS)

STATUS="v536_CAMS_2040_2050_original_frozen_2_3_5_day_event_sensitivity"

def gate(ref,year,frozen):
    if (ref.get("year")!=year or ref.get("source_model")!="CAMS-CSM1-0"
        or ref.get("ssp_fitted") is not False
        or ref.get("checkpoint_sha")!={f"{m}_weight{w}_seed{s}":v["sha256"]
            for (m,w,s),v in frozen["checkpoints"].items()}
        or len(ref.get("checkpoint_sha",{}))!=18
        or set(ref.get("scenarios",{}))!=set(SSPS)):
        raise ValueError("CAMS three-day original archive or Historical frozen SHA mismatch")
    for s in SSPS:
        data=ref["scenarios"][s]
        source=data.get("source",{})
        if (data.get("n_pairs")!=91
            or source.get("n_bounded_regional_native_days_read")!=92
            or source.get("n_source_days")!=91
            or not str(source.get("target_first","")).startswith(f"{year}-06-02")
            or not str(source.get("target_last","")).startswith(f"{year}-08-31")
            or set(data.get("models",{}))!={"fno","unet"}):
            raise ValueError("CAMS 2040/2050 3-day reference native date/grid mismatch")

def analyze(frozen,manifest,reference_2040,reference_2050):
    if len(frozen.get("checkpoints",{}))!=18:
        raise ValueError("18 original CAMS Historical-only frozen checkpoints required")
    for year,ref in ((2040,reference_2040),(2050,reference_2050)):
        gate(ref,year,frozen)
    row=manifest_for_CAMS(manifest,frozen)
    import torch
    torch.set_num_threads(1)
    sha={f"{m}_weight{w}_seed{s}":v["sha256"]
        for (m,w,s),v in frozen["checkpoints"].items()}
    out={
       "status":STATUS,"source_model":"CAMS-CSM1-0",
       "years":list(YEARS),"scenarios":list(SSPS),"durations":list(DURATIONS),
       "checkpoint_sha":sha,"n_original_frozen_checkpoints":18,
       "ssp_fitted":False,"independent_GCM_count":1,
       "all_original_three_day_reference_scores_reproduced":False,
       "records":[],
       "limitations":[
           "Single CAMS original native member, not a new climate-model ensemble",
           "2040 and 2050 summers, SSP245 and SSP585, remain correlated within the same GCM",
           "2-day and 5-day diagnostic definitions are exploratory, after inspecting 2050 3-day results",
           "Daily mean TAS simulated future fields, not tasmax, observed exposure, or tracked heatwave objects",
           "91-day summer-window boundary censoring, no spatial or temporal pseudoreplicate significance",
           "Training checkpoints and Historical-only P95 fixed before every future SSP read"]}
    for year,reference in ((2040,reference_2040),(2050,reference_2050)):
        for scenario in SSPS:
            X,Y,months,source=real_future_91(row[scenario]["store"],scenario,frozen,year=year)
            if len(X)!=91 or X.shape!=Y.shape or X.shape[1]!=1:
                raise ValueError("CAMS missing original 91 native Kelvin targets")
            baseline={d:event(X,Y,months,frozen["p95"],frozen["lat"],d) for d in DURATIONS}
            for d in DURATIONS:
                if d==3:
                    for metric in ("event_day_recall","event_day_precision"):
                        check_match(baseline[d][metric],
                            reference["scenarios"][scenario]["persistence"][metric],
                            f"CAMS/{year}/{scenario}/Persistence/{metric}")
                out["records"].append({
                    "year":year,"scenario":scenario,"min_days":d,
                    "model":"persistence","historical_multiplier":None,
                    "event_day_recall":baseline[d]["event_day_recall"],
                    "event_day_precision":baseline[d]["event_day_precision"],
                    "observed_event_day_area_weight":baseline[d]["observed_event_day_area_weight"],
                    "predicted_event_day_area_weight":baseline[d]["predicted_event_day_area_weight"]})
            for weight in WEIGHTS:
                by_duration={d:[] for d in DURATIONS}
                for seed in SEEDS:
                    forecast=fixed_prediction(frozen,"fno",weight,seed,X)
                    for d in DURATIONS:
                        by_duration[d].append(event(forecast,Y,months,frozen["p95"],frozen["lat"],d))
                for d in DURATIONS:
                    recall=summarize(by_duration[d],"event_day_recall")
                    precision=summarize(by_duration[d],"event_day_precision")
                    if d==3:
                        old=reference["scenarios"][scenario]["models"]["fno"][str(weight)]
                        check_match(recall,old["event_day_recall"]["mean"],
                                    f"CAMS/{year}/{scenario}/FNO/{weight}/Recall")
                        check_match(precision,old["event_day_precision"]["mean"],
                                    f"CAMS/{year}/{scenario}/FNO/{weight}/Precision")
                    out["records"].append({
                        "year":year,"scenario":scenario,"min_days":d,
                        "model":"fno","historical_multiplier":1+weight,
                        "event_day_recall":recall,"event_day_precision":precision,
                        "observed_event_day_area_weight":by_duration[d][0]["observed_event_day_area_weight"],
                        "predicted_event_day_area_weight":float(np.mean([
                            v["predicted_event_day_area_weight"] for v in by_duration[d]]))})
    if len(out["records"])!=36:raise ValueError("Incomplete original 36 CAMS duration rows")
    out["all_original_three_day_reference_scores_reproduced"]=True
    return out

def main():
    p=argparse.ArgumentParser()
    for k in ("historical","metadata","arm-dir","reference-2040","reference-2050","output"):
        p.add_argument("--"+k,required=True)
    a=vars(p.parse_args());target=Path(a["output"])
    if target.exists():raise FileExistsError("Do not overwrite CAMS duration evidence")
    frozen=inspect_frozen(a["historical"],a["metadata"],a["arm_dir"])
    catalog,sha=read_catalog()
    manifest=select_pairs(catalog,sha,max_models=12)
    report=analyze(frozen,manifest,json.loads(Path(a["reference_2040"]).read_text()),
                   json.loads(Path(a["reference_2050"]).read_text()))
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":STATUS,"records":len(report["records"]),
       "original_three_day_reproduced":report["all_original_three_day_reference_scores_reproduced"],
       "independent_GCM_count":1},indent=2))
if __name__=="__main__":main()
