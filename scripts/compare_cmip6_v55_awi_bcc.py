"""Compare frozen AWI and BCC future-domain contrasts without pooling GCM pixels.

Input: actual verified full v4.8 AWI and v5.4 BCC JSON reports.
No refitting, model selection or pseudo-replicated gridcell significance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

SCENARIOS=("ssp245","ssp585")
MODELS=("fno","unet")
WEIGHTS=("0","1","3")


def extract(model,raw,scenario,scope):
    if model=="AWI":
        p=raw["scenarios"][scenario]["groups"][scope]
        persistence=p["persistence"]
    elif model=="BCC":
        p=raw["scenario_results"][scenario]["scores"][scope]
        persistence=p["persistence"]
    else:
        raise ValueError("Unknown GCM")
    return p,persistence["weighted"]["rmse"],persistence["local_p95" if model=="AWI" else "monthly_hot"]["hot_recall"]


def compare(awi,bcc):
    if (awi["status"]!="first_frozen_historical_to_sparse_SSP_forecast_evaluation"
        or bcc["status"]!="first_independent_GCM_BCC_Historical_to_sparse_SSP_frozen_forecast"):
        raise ValueError("Require successful frozen source-model SSP reports")
    if (awi["ssp_used_for_fitting_or_selection"] is not False or
        bcc["ssp_used_for_fitting_or_checkpoint_selection"] is not False):
        raise ValueError("Future data may not be used for fitted model selection")
    if (awi["historical_npz_sha256"]!="ac0a354032836c5de5463b122d26fad2270fd5ba2ebb48f43074105c8ce7b914"
        or bcc["historical_source_sha256"]!="a23260e8f14a23fb57483038bcd6ca03ba74c7944a48ba29d62bcc5bd354f452"):
        raise ValueError("Unverified source model Historical artifacts")
    if awi["future_years"] != [2030,2050,2080] or bcc["future_years"] != [2030,2050,2080]:
        raise ValueError("Future year plan differs")
    if awi["future_month_days"] != [8,20] or bcc["future_month_days"] != [8,20]:
        raise ValueError("SSP sampled dates differ")
    if len(awi["checkpoint_sha256"])!=18 or len(bcc["checkpoint_sha"])!=18:
        raise ValueError("All 18 historical checkpoints required in each GCM")
    rows=[]
    for scope in ("all","2030","2050","2080"):
        for ssp in SCENARIOS:
            for source,report in (("AWI",awi),("BCC",bcc)):
                p,persist,precall=extract(source,report,ssp,scope)
                if p["n_pairs"]!=(72 if scope=="all" else 24):
                    raise ValueError("Incomplete matched scenario-year sample")
                for name in MODELS:
                    for weight in WEIGHTS:
                        m=p["models"][name][weight]
                        rmse=m["weighted_rmse_K"]["mean"]
                        recall=m["local_p95_recall"]["mean"]
                        prec=m["local_p95_precision"]["mean"]
                        if any(x is None for x in (rmse,recall,prec,persist,precall)) or persist<=0:
                            raise ValueError("Incomplete model comparison")
                        rows.append({
                            "source_gcm":source,"ssp":ssp,"scope":scope,
                            "model":name,"training_tail_weight":int(weight),
                            "training_hot_multiplier":int(weight)+1,
                            "n_source_target_pairs":p["n_pairs"],
                            "weighted_rmse_K":rmse,
                            "persistence_weighted_rmse_K":persist,
                            "rmse_improvement_fraction_vs_own_persistence":1-rmse/persist,
                            "local_p95_recall":recall,
                            "persistence_local_p95_recall":precall,
                            "recall_delta_vs_own_persistence":recall-precall,
                            "local_p95_precision":prec,
                        })
    if len(rows)!=2*2*4*2*3:
        raise ValueError("Incomplete cross-GCM contrast")
    return {
        "status":"paired_within_GCM_projected_temperature_forecast_contrasts_no_pooled_significance",
        "models":["AWI-CM-1-1-MR","BCC-CSM2-MR"],
        "ssp_scenarios":list(SCENARIOS),
        "test_source_years":[2030,2050,2080],
        "rows":rows,
        "study_limits":[
            "Compare within-GCM changes relative to each native Persistence baseline, not pooled gridcell errors",
            "AWI proleptic-gregorian 26x27 and BCC noleap 23x23 differ in geometry/calendar",
            "Two model members do not represent full CMIP6 ensemble or independent future observations",
            "Sparse twice-monthly projected scenario samples cannot quantify heatwave durations",
            "Historical-only model checkpoints selected using each GCM's Historical validation data",
            "No causal claim or significance from pseudo-replicated spatial grid cells",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--awi",required=True)
    p.add_argument("--bcc",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError("Cannot overwrite comparison")
    report=compare(json.loads(Path(a.awi).read_text()),
                   json.loads(Path(a.bcc).read_text()))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"n_rows":len(report["rows"]),
                      "models":report["models"]},indent=2))


if __name__=="__main__":
    main()
