"""E1 v5.31 CAMS independent 2040 summer native calendar time holdout."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from scripts.evaluate_cmip6_v518_cams_frozen_ssp import inspect_frozen,manifest_for_CAMS
from scripts.audit_cmip6_catalog import read_catalog,select_pairs
from scripts.evaluate_cmip6_v520_cams_contiguous_summer import evaluate,STATUS as V520_STATUS

YEAR=2040
STATUS="v531_CAMS_2040_predeclared_contiguous_summer_temporal_holdout"

def verify(r):
    if (r.get("status")!=STATUS or r.get("source_model")!="CAMS-CSM1-0"
        or r.get("year")!=YEAR or r.get("ssp_fitted") is not False
        or len(r.get("checkpoint_sha",{}))!=18
        or set(r.get("scenarios",{}))!={"ssp245","ssp585"}):
        raise ValueError("CAMS 2040 provenance mismatch")
    for s,d in r["scenarios"].items():
        source=d.get("source",{})
        if (d.get("n_pairs")!=91
            or source.get("n_source_days")!=91
            or source.get("n_bounded_regional_native_days_read")!=92
            or not source.get("target_first","").startswith("2040-06-02")
            or not source.get("target_last","").startswith("2040-08-31")):
            raise ValueError("CAMS 2040 continuous native target-day violation")
        if set(d.get("models",{}))!={"fno","unet"}:
            raise ValueError("Incomplete CAMS 2040 architecture inventory")
        for name in ("fno","unet"):
            if set(d["models"][name])!={"0","1","3"}:
                raise ValueError("Incomplete CAMS 2040 frozen loss arms")
            for metrics in d["models"][name].values():
                if any(len(metrics[k]["individual"])!=3 for k in ("rmse_K","event_day_recall","event_day_precision")):
                    raise ValueError("Incomplete CAMS original seeds")
    return r

def run(frozen,manifest):
    if len(frozen.get("checkpoints",{}))!=18:
        raise ValueError("Require 18 original frozen CAMS Historical checkpoints before SSP access")
    r=evaluate(frozen,manifest,year=YEAR)
    if r.get("status")!=V520_STATUS:raise ValueError("Changed original CAMS evaluator")
    r["status"]=STATUS
    r["temporal_independence_from_prior_2050_pilot"]=True
    r["selection_policy"]="Report all original Historical-only frozen arms, no selection based on 2050 or 2040"
    r["limits"].append("2040 evaluation year fixed after exploratory 2050 results and before reading new 2040 data")
    return verify(r)

def main():
    p=argparse.ArgumentParser()
    for k in ("historical","metadata","arm-dir","output"):p.add_argument("--"+k,required=True)
    a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError("Do not overwrite independent 2040 CAMS source evidence")
    frozen=inspect_frozen(a.historical,a.metadata,a.arm_dir)
    cat,sha=read_catalog()
    manifest=select_pairs(cat,sha,max_models=12)
    r=run(frozen,manifest)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":STATUS,"year":YEAR,"scenarios":{
        s:{"n_pairs":v["n_pairs"],
           "persistence_event_day_recall":v["persistence"]["event_day_recall"],
           "fno_1x_event_day_recall":v["models"]["fno"]["0"]["event_day_recall"]["mean"],
           "fno_4x_event_day_recall":v["models"]["fno"]["3"]["event_day_recall"]["mean"]}
        for s,v in r["scenarios"].items()}},indent=2))
if __name__=="__main__":main()
