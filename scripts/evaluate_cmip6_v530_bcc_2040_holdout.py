"""E1 v5.30 BCC-CSM2-MR 2040 native noleap held-out summer, original checkpoints."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from scripts.evaluate_cmip6_bcc_v54_frozen_ssp import inspect_frozen_sources
from scripts.evaluate_cmip6_v523_bcc_contiguous_summer import evaluate,STATUS as V523_STATUS

STATUS="v530_BCC_independent_2040_contiguous_summer_holdout"
YEAR=2040
def verify(r):
    if (r.get("status")!=STATUS or r.get("year")!=YEAR
        or r.get("source_model")!="BCC-CSM2-MR"
        or r.get("ssp_fitted") is not False
        or len(r.get("checkpoint_sha",{}))!=18
        or set(r.get("scenarios",{}))!={"ssp245","ssp585"}):
        raise ValueError("BCC frozen 2040 provenance violation")
    for s,sc in r["scenarios"].items():
        src=sc["source"]
        if (sc.get("n_pairs")!=91 or src.get("native_calendar") not in ("noleap","365_day")
            or src.get("native_grid_shape")!=[23,23]
            or src.get("n_bounded_regional_native_days_read")!=92
            or not str(src.get("first_target","")).startswith("2040-06-02")
            or not str(src.get("last_target","")).startswith("2040-08-31")):
            raise ValueError("BCC 2040 day/native grid violation")
        if set(sc.get("models",{}))!={"fno","unet"}:
            raise ValueError("Missing BCC frozen architecture")
        for model in ("fno","unet"):
            if set(sc["models"][model])!={"0","1","3"}:
                raise ValueError("Missing BCC frozen Historical loss arm")
            for arm in sc["models"][model].values():
                if any(len(arm[k]["individual"])!=3 for k in ("rmse_K","event_day_recall","event_day_precision")):
                    raise ValueError("Missing BCC original frozen seeds")
    return r
def evaluate_2040(frozen,manifest):
    if len(frozen.get("checkpoints",{}))!=18:
        raise ValueError("All 18 BCC Historical checkpoints required before SSP read")
    r=evaluate(frozen,manifest,year=YEAR)
    if r["status"]!=V523_STATUS:raise ValueError("BCC shared evaluator has changed")
    r["status"]=STATUS
    r["selection_policy"]="All original Historical frozen arms, no tuning or weight selection using 2040"
    r["temporal_independence_from_2050"]=True
    r["limitations"].append("2040 is a new temporal holdout selected after exploratory 2050 analysis; still one BCC member")
    return verify(r)
def main():
    ap=argparse.ArgumentParser()
    for k in ("historical","arm-dir","manifest","output"):ap.add_argument("--"+k,required=True)
    a=ap.parse_args()
    target=Path(a.output)
    if target.exists():raise FileExistsError("Do not overwrite native BCC 2040 results")
    frozen=inspect_frozen_sources(Path(a.historical),Path(a.arm_dir))
    r=evaluate_2040(frozen,json.loads(Path(a.manifest).read_text()))
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":r["status"],"year":r["year"],"scenarios":{
       s:{"n_pairs":v["n_pairs"],"persistence_event_day_recall":v["persistence"]["event_day_recall"],
       "fno_1x_event_day_recall":v["models"]["fno"]["0"]["event_day_recall"]["mean"],
       "fno_4x_event_day_recall":v["models"]["fno"]["3"]["event_day_recall"]["mean"]}
       for s,v in r["scenarios"].items()}},indent=2))
if __name__=="__main__":main()
