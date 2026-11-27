"""E1 v5.29: independent AWI native Gregorian 2040 summer frozen holdout.

Fixed before reading 2040 SSP: year 2040 (not 2050 validation summer),
two SSP scenarios, 91 consecutive 24-hour targets, identical Historical P95,
all 18 originally frozen checkpoints. No new training or model selection.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.evaluate_cmip6_v48_frozen_ssp import read_frozen_checkpoints
from scripts.evaluate_cmip6_v524_awi_contiguous_summer import evaluate,STATUS as V524_STATUS

STATUS="v529_AWI_predeclared_2040_independent_year_contiguous_summer"
YEAR=2040

def verified_frozen(historical,arm_dir):
    historical=Path(historical)
    frozen=read_frozen_checkpoints(Path(arm_dir),historical)
    if len(frozen["models"])!=18 or len(frozen["checkpoint_sha256_by_arm"])!=18:
        raise ValueError("Incomplete immutable AWI source checkpoint inventory")
    meta=json.loads(historical.with_suffix(".json").read_text())
    if (meta.get("status")!="v45_historical_dense_sparsely_sampled_training_ready_no_model_training"
        or meta.get("dataset_sha256")!=hashlib.sha256(historical.read_bytes()).hexdigest()
        or meta.get("NPZ_contains_only_historical") is not True
        or meta.get("ssp_test_data_read") is not False
        or meta.get("region_grid_shape")!=[26,27]
        or len(meta.get("longitude_east_bounds",[]))!=2
        or not np.allclose(meta.get("latitude_degrees",[]),frozen["latitude"],atol=1e-8,rtol=0)):
        raise ValueError("Original Historical metadata does not match verified AWI training NPZ")
    frozen["historical_lon_bounds"]=meta["longitude_east_bounds"]
    return frozen

def verify_output(report):
    if (report.get("status")!=STATUS or report.get("year")!=YEAR
        or report.get("ssp_fitted") is not False
        or len(report.get("checkpoint_sha",{}))!=18
        or set(report.get("scenarios",{}))!={"ssp245","ssp585"}):
        raise ValueError("Missing complete AWI frozen 2040 summer output")
    for s,sc in report["scenarios"].items():
        m=sc["source"]
        if (sc.get("n_pairs")!=91
            or m.get("native_calendar")!="proleptic_gregorian"
            or m.get("native_grid_shape")!=[26,27]
            or m.get("n_bounded_regional_native_days_read")!=92
            or not str(m.get("first_target","")).startswith("2040-06-02")
            or not str(m.get("last_target","")).startswith("2040-08-31")):
            raise ValueError("Wrong 2040 native Gregorian evaluation days or geometry")
        if set(sc["models"])!={"fno","unet"}:raise ValueError("Missing neural family")
        for model in ("fno","unet"):
            if set(sc["models"][model])!={"0","1","3"}:raise ValueError("Missing frozen training loss arm")
            for w,x in sc["models"][model].items():
                for metric in ("rmse_K","event_day_recall","event_day_precision"):
                    z=x[metric]
                    if len(z["individual"])!=3:raise ValueError("Missing original seeds")
    return report

def run(frozen,manifest):
    if len(frozen.get("models",{}))!=18:
        raise ValueError("Historical 18 frozen checkpoints required before SSP read")
    raw=evaluate(frozen,manifest,year=YEAR)
    if raw.get("status")!=V524_STATUS:raise ValueError("Unexpected AWI 2050-compatible evaluator status")
    raw["status"]=STATUS
    raw["selection_policy"]="Original Historical-only selection, all six arms per SSP; no 2040 checkpoint selection"
    raw["temporal_independence_from_prior_2050_event_analysis"]=True
    raw["limitations"].append("2040 explicitly chosen before this separate 2040 score read; not another independent GCM member")
    return verify_output(raw)

def main():
    p=argparse.ArgumentParser()
    for k in ("historical","arm-dir","manifest","output"):p.add_argument("--"+k,required=True)
    a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError("Do not overwrite 2040 held-out output")
    frozen=verified_frozen(a.historical,a.arm_dir)
    report=run(frozen,json.loads(Path(a.manifest).read_text()))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"year":YEAR,
      "scenarios":{s:{"n_pairs":v["n_pairs"],
         "persistence_event_day_recall":v["persistence"]["event_day_recall"],
         "fno_1x_event_day_recall":v["models"]["fno"]["0"]["event_day_recall"]["mean"],
         "fno_4x_event_day_recall":v["models"]["fno"]["3"]["event_day_recall"]["mean"]}
         for s,v in report["scenarios"].items()}},indent=2))
if __name__=="__main__":main()
