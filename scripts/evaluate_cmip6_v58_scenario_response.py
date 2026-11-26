"""E1 v5.8 frozen scenario-response diagnostic (not forced-warming attribution).

For matched GCM/year/calendar dates, model scenario-response error =
forecast_bias_ssp585 - forecast_bias_ssp245. The two simulations have
different internal variability; this is a *sampled response-error contrast*.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from scripts.evaluate_cmip6_v57_sampled_change import (
    validate_source, extract_bias, YEARS, MODELS, TAIL_WEIGHTS,
)

def summarize(awi, bcc):
    outputs = []
    for gcm, report in (("AWI",awi),("BCC",bcc)):
        validate_source(gcm, report)
        for year in YEARS:
            for model, weight in (("persistence",None),) + tuple(
                (m,w) for m in MODELS for w in TAIL_WEIGHTS):
                b245 = extract_bias(gcm,report,"ssp245",year,model,weight)
                b585 = extract_bias(gcm,report,"ssp585",year,model,weight)
                if len(b245)!=len(b585):
                    raise ValueError("Unmatched frozen seed sets")
                differences=[a-b for a,b in zip(b585,b245)]
                mean=sum(differences)/len(differences)
                outputs.append({
                    "gcm":gcm,"year":year,"predictor":model,
                    "historical_train_tail_weight":None if weight is None else int(weight),
                    "scenario_pair":"ssp585_minus_ssp245",
                    "date_pairs_per_scenario":24,
                    "per_seed_scenario_response_errors_K":differences,
                    "mean_scenario_response_error_K":mean,
                    "absolute_mean_scenario_response_error_K":abs(mean),
                    "mean_absolute_seed_response_error_K":sum(map(abs,differences))/len(differences),
                    "ssp245_forecast_bias_K":b245,
                    "ssp585_forecast_bias_K":b585
                })
    if len(outputs)!=42:
        raise ValueError("Expected all 42 source-year-predictor comparisons")
    baselines={(r["gcm"],r["year"]):r for r in outputs if r["predictor"]=="persistence"}
    for r in outputs:
        b=baselines[r["gcm"],r["year"]]["absolute_mean_scenario_response_error_K"]
        r["persistence_abs_mean_response_error_K"]=b
        r["absolute_mean_error_difference_vs_persistence_K"]=r["absolute_mean_scenario_response_error_K"]-b
    return {
        "status":"v58_sparse_ssp585_minus_ssp245_scenario_response_error_no_forced_signal_attribution",
        "source_runs":{"AWI":45864317174,"BCC":45872811683},
        "n_source_gcms":2,"members_per_gcm":1,
        "matched_dates":{"days_per_month":[8,20],"pairs_each_year":24},
        "rows":outputs,
        "limitations":[
            "Scenario difference represents sparse sampled daily temperature simulations; not forced climate response",
            "No scenario-response truth from future observations exists",
            "Different scenario runs can contain different internal climate variability",
            "Absolute mean bias contrast is distinct from mean absolute per-seed error",
            "Two source GCMs with one member each prevent ensemble-level uncertainty statements",
            "Results cannot be used to select model architecture, tail weight or checkpoint"
        ]
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--awi",required=True)
    p.add_argument("--bcc",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    path=Path(a.output)
    if path.exists():
        raise FileExistsError("Immutable experiment output already exists")
    result=summarize(json.loads(Path(a.awi).read_text()),
                     json.loads(Path(a.bcc).read_text()))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"row_count":len(result["rows"])},indent=2))

if __name__=="__main__":
    main()
