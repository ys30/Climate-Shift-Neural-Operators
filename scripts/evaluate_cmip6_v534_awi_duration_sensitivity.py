"""E1 v5.34 — AWI fixed-checkpoint 2/3/5-day event sensitivity.

Read native historical geometry and 18 SHA-audited original Historical checkpoints
before future SSP access. Original 2040 and 2050 source days; no new model fit.
For each pixel use the exact Historical-training month-local TAS P95, then
mark full runs of >=2/3/5 days, including 91-day window edge censoring.
Require exact reproduction of independently archived 3-day reference metrics.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from scripts.evaluate_cmip6_v529_awi_2040_contiguous_holdout import verified_frozen
from scripts.evaluate_cmip6_v524_awi_contiguous_summer import read_awi_91,predict
from scripts.evaluate_cmip6_v520_cams_contiguous_summer import run_mask
from scripts.evaluate_cmip6_v48_frozen_ssp import manifest_entry

STATUS="v534_AWI_2040_2050_two_three_five_day_original_frozen_sensitivity"
YEARS=(2040,2050)
SSPS=("ssp245","ssp585")
DURATIONS=(2,3,5)
SEEDS=(11,22,33)
WEIGHTS=(0,3)

def event(pred,target,months,p95,latitude,min_days):
    if pred.shape!=target.shape or pred.ndim!=4 or pred.shape[0]!=91 or pred.shape[1]!=1:
        raise ValueError("Exactly 91 aligned original native one-day targets required")
    if np.asarray(months).shape!=(91,) or not set(np.asarray(months).tolist()).issubset({6,7,8}):
        raise ValueError("Invalid native summer target month")
    if p95.shape!=(12,pred.shape[2],pred.shape[3]):
        raise ValueError("Historical monthly P95 spatial geometry mismatch")
    if np.asarray(latitude).shape!=(pred.shape[2],):
        raise ValueError("Latitude geometry mismatch")
    if min_days not in DURATIONS:raise ValueError("Duration must be 2, 3, or 5")
    truth=run_mask(target[:,0]>p95[np.asarray(months)-1],min_length=min_days)
    fcst=run_mask(pred[:,0]>p95[np.asarray(months)-1],min_length=min_days)
    weight=np.broadcast_to(np.cos(np.deg2rad(latitude))[None,:,None],truth.shape)
    observed=float((truth*weight).sum())
    forecast=float((fcst*weight).sum())
    overlap=float((truth*fcst*weight).sum())
    return {
        "observed_event_day_area_weight":observed,
        "predicted_event_day_area_weight":forecast,
        "matched_event_day_area_weight":overlap,
        "event_day_recall":overlap/observed if observed>0 else None,
        "event_day_precision":overlap/forecast if forecast>0 else None,
        "boundary_censoring":"Event runs touching either summer edge may continue outside the evaluated window"}

def reference_gate(report,year,sha):
    if (report.get("year")!=year or report.get("ssp_fitted") is not False
        or report.get("checkpoint_sha")!=sha
        or set(report.get("scenarios",{}))!=set(SSPS)):
        raise ValueError("Source 3-day reference changed or checkpoint SHA drifted")

def check_match(new,old,label,tolerance=2e-7):
    if (new is None)!=(old is None) or (new is not None and
        not math.isclose(new,old,abs_tol=tolerance,rel_tol=0)):
        raise ValueError(f"Original archived 3-day event score differs for {label}: {new} != {old}")

def summarize(entries,field):
    values=[q[field] for q in entries]
    if len(values)!=3:raise ValueError("Missing original seeds")
    if all(v is None for v in values):return None
    if any(v is None for v in values):raise ValueError("Partial undefined seed score")
    return float(np.mean(values))

def analyze(frozen,manifest,reference_2040,reference_2050):
    if len(frozen.get("models",{}))!=18 or len(frozen.get("checkpoint_sha256_by_arm",{}))!=18:
        raise ValueError("All 18 SHA audited Historical-only original checkpoints required")
    sha=frozen["checkpoint_sha256_by_arm"]
    for year,ref in ((2040,reference_2040),(2050,reference_2050)):
        reference_gate(ref,year,sha)
    row=manifest_entry(manifest)
    out={"status":STATUS,
      "source_model":"AWI-CM-1-1-MR","years":list(YEARS),
      "scenarios":list(SSPS),"durations":list(DURATIONS),
      "checkpoint_sha":sha,"ssp_fitted":False,
      "n_original_frozen_checkpoints":18,
      "historical_only_month_local_threshold":True,
      "independent_GCM_count":1,
      "records":[],
      "limitations":[
        "This is a sensitivity experiment on ONE GCM member, not 3 independent models",
        "2040 and 2050 years and two SSPs are correlated within the original AWI GCM",
        "2/3/5-day definitions were considered after inspecting original 3-day results: exploratory",
        "91 target days and window-edge event censoring; event-duration counts beyond window unavailable",
        "This assesses event-day overlap, not tracked heatwave objects, durations, or tasmax exposure",
        "No re-fitting on future SSP; no formal independent p-values"]}
    import torch
    torch.set_num_threads(1)
    for year,ref in ((2040,reference_2040),(2050,reference_2050)):
        for scenario in SSPS:
            X,Y,months,meta=read_awi_91(row[scenario]["store"],scenario,frozen,year=year)
            assert X.shape==Y.shape==(91,1,26,27)
            base={n:event(X,Y,months,frozen["local_p95"],frozen["latitude"],n) for n in DURATIONS}
            for n in DURATIONS:
                if n==3:
                    for metric in ("event_day_recall","event_day_precision"):
                        check_match(base[n][metric],ref["scenarios"][scenario]["persistence"][metric],
                                    f"{year}/{scenario}/persistence/{metric}")
                out["records"].append({
                    "year":year,"scenario":scenario,"min_days":n,
                    "model":"persistence","historical_multiplier":None,
                    "event_day_recall":base[n]["event_day_recall"],
                    "event_day_precision":base[n]["event_day_precision"],
                    "observed_event_day_area_weight":base[n]["observed_event_day_area_weight"],
                    "predicted_event_day_area_weight":base[n]["predicted_event_day_area_weight"]})
            for model in ("fno",):
                for weight in WEIGHTS:
                    per_seed={n:[] for n in DURATIONS}
                    for seed in SEEDS:
                        forecast=predict(frozen,model,weight,seed,X)
                        for n in DURATIONS:
                            per_seed[n].append(event(forecast,Y,months,frozen["local_p95"],frozen["latitude"],n))
                    for n in DURATIONS:
                        recall=summarize(per_seed[n],"event_day_recall")
                        precision=summarize(per_seed[n],"event_day_precision")
                        if n==3:
                            reference=ref["scenarios"][scenario]["models"][model][str(weight)]
                            check_match(recall,reference["event_day_recall"]["mean"],f"{year}/{scenario}/{model}/{weight}/Recall")
                            check_match(precision,reference["event_day_precision"]["mean"],f"{year}/{scenario}/{model}/{weight}/Precision")
                        out["records"].append({
                            "year":year,"scenario":scenario,"min_days":n,
                            "model":model,"historical_multiplier":1+weight,
                            "event_day_recall":recall,
                            "event_day_precision":precision,
                            "observed_event_day_area_weight":per_seed[n][0]["observed_event_day_area_weight"],
                            "predicted_event_day_area_weight":float(np.mean([v["predicted_event_day_area_weight"] for v in per_seed[n]]))})
    if len(out["records"])!=36:raise ValueError("Missing 2 years × 2 SSP × 3 durations × 3 models")
    out["all_original_three_day_reference_scores_reproduced"]=True
    return out

def main():
    p=argparse.ArgumentParser()
    for x in ("historical","arm-dir","manifest","reference-2040","reference-2050","output"):
        p.add_argument("--"+x,required=True)
    a=vars(p.parse_args());output=Path(a["output"])
    if output.exists():raise FileExistsError("Do not overwrite duration-sensitivity evidence")
    f=verified_frozen(a["historical"],a["arm_dir"])
    manifest=json.loads(Path(a["manifest"]).read_text())
    ref40=json.loads(Path(a["reference_2040"]).read_text())
    ref50=json.loads(Path(a["reference_2050"]).read_text())
    out=analyze(f,manifest,ref40,ref50)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":STATUS,"records":len(out["records"]),
                      "3day_original_reference_reproduced":True,
                      "reference_GCM_count":1},indent=2))
if __name__=="__main__":main()
