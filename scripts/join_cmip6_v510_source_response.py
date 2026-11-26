"""E1 v5.10 audited join of native source SSP contrasts and frozen forecast errors.

Only original v5.9 physical target means and v5.8 forecast-bias
differences. Identity within a GCM/year:
  predicted(SSP585-SSP245) = actual_simulated_contrast + error.
No model refit, no cross-GCM grid pooling and no forced-trend inference.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

GCMS=("AWI","BCC")
SSPS=("ssp245","ssp585")
YEARS=(2030,2050,2080)
MODELS=("persistence","fno","unet")
WEIGHTS={"persistence":(None,),"fno":(0,1,3),"unet":(0,1,3)}

def finite(x):
    if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x):
        raise ValueError("Nonfinite or invalid temperature response")
    return float(x)

def build(source, error):
    if source.get("status")!="v59_sparse_CMIP6_source_only_temperature_reference":
        raise ValueError("Require verified v5.9 source-only input")
    if error.get("status")!="v58_sparse_ssp585_minus_ssp245_scenario_response_error_no_forced_signal_attribution":
        raise ValueError("Require verified frozen v5.8 scenario-response input")
    if source.get("historical_reference")!=[2005,2014] or source.get("future_years")!=list(YEARS):
        raise ValueError("Historical/future reference changed")
    if source.get("future_source_days")!=[8,20] or error.get("matched_dates")!={"days_per_month":[8,20],"pairs_each_year":24}:
        raise ValueError("Unmatched calendar days")
    physical={}
    for row in source.get("rows",[]):
        key=(row["gcm"],row["scenario"],row["year"])
        if key in physical or key not in {(g,s,y) for g in GCMS for s in SSPS for y in YEARS}:
            raise ValueError("Duplicate or unexpected source scenario-year")
        if row["n_historical_pairs"]!=240 or row["n_future_pairs"]!=24:
            raise ValueError("Changed sample count")
        h=finite(row["historical_regional_mean_K"])
        t=finite(row["projected_sparse_regional_mean_K"])
        change=finite(row["sampled_change_vs_historical_K"])
        if not math.isclose(t-h,change,abs_tol=1e-5):
            raise ValueError("Source reference temperature change inconsistency")
        if not 150<h<350 or not 150<t<350:
            raise ValueError("Physically invalid Kelvin mean")
        physical[key]=(h,t,change)
    expected={(g,s,y) for g in GCMS for s in SSPS for y in YEARS}
    if set(physical)!=expected:
        raise ValueError("Incomplete v5.9 physical source values")
    for g in GCMS:
        for y in YEARS:
            h245=physical[(g,"ssp245",y)][0]
            h585=physical[(g,"ssp585",y)][0]
            if not math.isclose(h245,h585,abs_tol=1e-6):
                raise ValueError("Scenarios do not share same historical GCM reference")
    indexed={}
    allowed={(g,y,m,w) for g in GCMS for y in YEARS for m in MODELS for w in WEIGHTS[m]}
    for row in error.get("rows",[]):
        key=(row["gcm"],row["year"],row["predictor"],row["historical_train_tail_weight"])
        if key in indexed or key not in allowed:
            raise ValueError("Duplicate or unexpected predictor response row")
        if row.get("scenario_pair")!="ssp585_minus_ssp245" or row.get("date_pairs_per_scenario")!=24:
            raise ValueError("Wrong frozen SSP pair or date count")
        bias=finite(row["mean_scenario_response_error_K"])
        arr=row["per_seed_scenario_response_errors_K"]
        if len(arr)!=(1 if row["predictor"]=="persistence" else 3):
            raise ValueError("Missing individual seed result")
        if not math.isclose(sum(finite(v) for v in arr)/len(arr),bias,abs_tol=1e-8):
            raise ValueError("Reported scenario error not supported by seeds")
        indexed[key]=bias
    if set(indexed)!=allowed:
        raise ValueError("Incomplete v5.8 model response table")
    results=[]
    for g in GCMS:
        for y in YEARS:
            source_delta=(physical[(g,"ssp585",y)][1]-physical[(g,"ssp245",y)][1])
            for m in MODELS:
                for w in WEIGHTS[m]:
                    bias=indexed[g,y,m,w]
                    results.append({
                        "gcm":g,"year":y,"predictor":m,"historical_tail_weight":w,
                        "source_simulated_ssp585_minus_245_K":source_delta,
                        "forecast_ssp585_minus_245_K":source_delta+bias,
                        "forecast_response_error_K":bias,
                        "absolute_forecast_response_error_K":abs(bias),
                        "n_dates_per_scenario":24
                    })
    if len(results)!=42:
        raise ValueError("Incomplete cross-source/model records")
    return {"status":"v510_verified_join_of_sparse_CMIP6_source_and_frozen_forecast_scenario_response",
        "upstream_runs":{"v58":45878008075,"v59":45891536056},
        "model_selection_performed":False,"n_source_gcms":2,"rows":results,
        "limitations":[
            "Derived from 24 sparse calendar-day target pairs per SSP/year; not a forced-warming estimate",
            "No independent future observations; both sources are CMIP6 model simulations",
            "No hypothetical independent spatial samples or significance testing",
            "No SSP-based checkpoint/weight optimization",
            "Historical 2005-2014 reference was previously used for neural checkpoint validation",
            "Derived predicted scenario difference is a regional mean only, not spatial pattern fidelity"
        ]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--error",required=True)
    p.add_argument("--json-output",required=True)
    p.add_argument("--csv-output",required=True)
    args=p.parse_args()
    out=Path(args.json_output);csv_out=Path(args.csv_output)
    if out.exists() or csv_out.exists():
        raise FileExistsError("Never overwrite versioned experiment")
    report=build(json.loads(Path(args.source).read_text()),json.loads(Path(args.error).read_text()))
    out.parent.mkdir(parents=True,exist_ok=True)
    csv_out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    with csv_out.open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(report["rows"][0]))
        w.writeheader();w.writerows(report["rows"])
    print(json.dumps({"status":report["status"],"rows":len(report["rows"])},indent=2))

if __name__=="__main__":
    main()
