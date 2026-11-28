"""E1 v5.35 BCC-CSM2-MR native noleap 2040/2050 2/3/5-day hot-event sensitivity.

Reuses exact AWI v5.34 event-day metric, but BCC's own native noleap
calendar, Historical-local monthly tas P95, 23x23 grid and SHA-audited
18 Historical-only frozen checkpoints. Reproduces all original archived
3-day scores exactly for both years/SSPs before accepting new thresholds.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.evaluate_cmip6_bcc_v54_frozen_ssp import inspect_frozen_sources
from scripts.evaluate_cmip6_v523_bcc_contiguous_summer import read_bcc_91,predict
from scripts.audit_cmip6_bcc_regional_pairs import get_model
from scripts.evaluate_cmip6_v534_awi_duration_sensitivity import (
    event,check_match,summarize,YEARS,SSPS,DURATIONS,SEEDS,WEIGHTS)
STATUS="v535_BCC_2040_2050_original_frozen_two_three_five_day_event_sensitivity"

def gate(ref,year,frozen):
    if (ref.get("year")!=year or ref.get("source_model")!="BCC-CSM2-MR"
        or ref.get("ssp_fitted") is not False
        or ref.get("checkpoint_sha")!=frozen["checkpoint_sha_by_model"]
        or set(ref.get("scenarios",{}))!=set(SSPS)):
        raise ValueError("BCC three-day source provenance or original checkpoint SHA mismatch")
    for s in SSPS:
        m=ref["scenarios"][s]["source"]
        if (ref["scenarios"][s].get("n_pairs")!=91
            or m.get("native_grid_shape")!=[23,23]
            or m.get("native_calendar") not in ("noleap","365_day")
            or m.get("n_bounded_regional_native_days_read")!=92):
            raise ValueError("BCC original 3-day benchmark changes source/date/grid")

def analyze(frozen,manifest,reference_2040,reference_2050):
    if (len(frozen.get("checkpoints",{}))!=18
        or len(frozen.get("checkpoint_sha_by_model",{}))!=18):
        raise ValueError("Require all 18 original BCC Historical-only frozen checkpoints")
    for year,ref in ((2040,reference_2040),(2050,reference_2050)):
        gate(ref,year,frozen)
    row=get_model(manifest)
    import torch
    torch.set_num_threads(1)
    out={"status":STATUS,"source_model":"BCC-CSM2-MR",
         "years":list(YEARS),"scenarios":list(SSPS),"durations":list(DURATIONS),
         "checkpoint_sha":frozen["checkpoint_sha_by_model"],
         "n_original_frozen_checkpoints":18,"ssp_fitted":False,
         "independent_GCM_count":1,"all_original_three_day_reference_scores_reproduced":False,
         "records":[],"limitations":[
            "One original BCC native noleap climate-model member; years and SSP are correlated",
            "All 18 frozen checkpoints and local monthly Historical-only P95, no 2040/2050 fitting",
            "2 and 5 day duration sensitivity is exploratory after original 3-day findings",
            "Day-aligned pixel event-day overlap, not duration of tracked climate heatwave objects",
            "Native simulated daily mean tas, not observed weather or tasmax",
            "Window-boundary events censored and no spatial/temporal pseudoreplicate significance"]}
    for year,reference in ((2040,reference_2040),(2050,reference_2050)):
        for scenario in SSPS:
            X,Y,months,source=read_bcc_91(row[scenario]["store"],scenario,frozen,year=year)
            if X.shape!=Y.shape!=(91,1,23,23):
                raise ValueError("Changed BCC native bounded 91-day region")
            base={d:event(X,Y,months,frozen["p95"],frozen["lat"],d) for d in DURATIONS}
            for d in DURATIONS:
                if d==3:
                    for metric in ("event_day_recall","event_day_precision"):
                        check_match(base[d][metric],
                            reference["scenarios"][scenario]["persistence"][metric],
                            f"BCC/{year}/{scenario}/Persistence/{metric}")
                out["records"].append({
                    "year":year,"scenario":scenario,"min_days":d,
                    "model":"persistence","historical_multiplier":None,
                    "event_day_recall":base[d]["event_day_recall"],
                    "event_day_precision":base[d]["event_day_precision"],
                    "observed_event_day_area_weight":base[d]["observed_event_day_area_weight"],
                    "predicted_event_day_area_weight":base[d]["predicted_event_day_area_weight"]})
            for weight in WEIGHTS:
                individual={d:[] for d in DURATIONS}
                for seed in SEEDS:
                    fc=predict(frozen,"fno",weight,seed,X)
                    for d in DURATIONS:
                        individual[d].append(event(fc,Y,months,frozen["p95"],frozen["lat"],d))
                for d in DURATIONS:
                    recall=summarize(individual[d],"event_day_recall")
                    precision=summarize(individual[d],"event_day_precision")
                    if d==3:
                        ref=reference["scenarios"][scenario]["models"]["fno"][str(weight)]
                        check_match(recall,ref["event_day_recall"]["mean"],
                            f"BCC/{year}/{scenario}/FNO/{weight}/Recall")
                        check_match(precision,ref["event_day_precision"]["mean"],
                            f"BCC/{year}/{scenario}/FNO/{weight}/Precision")
                    out["records"].append({
                        "year":year,"scenario":scenario,"min_days":d,
                        "model":"fno","historical_multiplier":1+weight,
                        "event_day_recall":recall,"event_day_precision":precision,
                        "observed_event_day_area_weight":individual[d][0]["observed_event_day_area_weight"],
                        "predicted_event_day_area_weight":float(np.mean([
                            v["predicted_event_day_area_weight"] for v in individual[d]]))})
    if len(out["records"])!=36:
        raise ValueError("Incomplete BCC 2 years x 2 SSP x 3 event durations x 3 methods")
    out["all_original_three_day_reference_scores_reproduced"]=True
    return out

def main():
    ap=argparse.ArgumentParser()
    for k in ("historical","arm-dir","manifest","reference-2040","reference-2050","output"):
        ap.add_argument("--"+k,required=True)
    a=vars(ap.parse_args());target=Path(a["output"])
    if target.exists():raise FileExistsError("Refuse to overwrite BCC 2/3/5-day diagnostic")
    f=inspect_frozen_sources(Path(a["historical"]),Path(a["arm_dir"]))
    report=analyze(f,json.loads(Path(a["manifest"]).read_text()),
                   json.loads(Path(a["reference_2040"]).read_text()),
                   json.loads(Path(a["reference_2050"]).read_text()))
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":STATUS,"records":len(report["records"]),
       "three_day_archived_reference_reproduced":True,
       "independent_GCM_count":report["independent_GCM_count"]},indent=2))
if __name__=="__main__":main()
