"""E1 v5.7: prospective diagnostic of *sampled* inter-year forecast change error.

Uses unchanged v4.8/v5.4 SSP score JSON; no new prediction, training,
checkpoint selection, or forced climate-trend claims. By linearity:
  (pred_year_B - pred_year_A) - (truth_year_B - truth_year_A)
    = forecast_bias_year_B - forecast_bias_year_A.
Annual summaries must have identical fixed days/month (24 date-pairs/year).
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

YEARS = (2030, 2050, 2080)
SSPS = ("ssp245", "ssp585")
MODELS = ("fno", "unet")
TAIL_WEIGHTS = ("0", "1", "3")
SEEDS = (11, 22, 33)
AWI_SHA = "ac0a354032836c5de5463b122d26fad2270fd5ba2ebb48f43074105c8ce7b914"
BCC_SHA = "a23260e8f14a23fb57483038bcd6ca03ba74c7944a48ba29d62bcc5bd354f452"

def validate_source(source, raw):
    if source == "AWI":
        if (raw.get("status") != "first_frozen_historical_to_sparse_SSP_forecast_evaluation"
                or raw.get("historical_npz_sha256") != AWI_SHA
                or raw.get("ssp_used_for_fitting_or_selection") is not False
                or len(raw.get("checkpoint_sha256", {})) != 18):
            raise ValueError("AWI source/checkpoint/fitting gate failed")
    elif source == "BCC":
        if (raw.get("status") != "first_independent_GCM_BCC_Historical_to_sparse_SSP_frozen_forecast"
                or raw.get("historical_source_sha256") != BCC_SHA
                or raw.get("ssp_used_for_fitting_or_checkpoint_selection") is not False
                or len(raw.get("checkpoint_sha", {})) != 18
                or raw.get("native_calendar") != "noleap"):
            raise ValueError("BCC source/checkpoint/calendar/fitting gate failed")
    else:
        raise ValueError("Unsupported source")
    if raw.get("future_years") != list(YEARS) or raw.get("future_month_days") != [8, 20]:
        raise ValueError("Changed future sampling schedule")

def group_for(source, raw, scenario, year):
    if source == "AWI":
        p = raw["scenarios"][scenario]["groups"][str(year)]
    else:
        p = raw["scenario_results"][scenario]["scores"][str(year)]
    if p["n_pairs"] != 24:
        raise ValueError("Each sampled future year requires exactly 24 dates")
    return p

def check_bias(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Nonfinite forecast bias")
    return float(value)

def extract_bias(source, raw, scenario, year, model, weight):
    p = group_for(source, raw, scenario, year)
    if model == "persistence":
        return [check_bias(p["persistence"]["weighted"]["bias"])]
    record = p["models"][model][weight]["weighted_bias_K"]
    individual = record["individual"]
    if len(individual) != len(SEEDS):
        raise ValueError("Require three frozen seeds, in original scoring order")
    values = [check_bias(x) for x in individual]
    if abs(sum(values)/len(values) - check_bias(record["mean"])) > 1e-5:
        raise ValueError("Seed biases do not reproduce stored mean")
    return values

def evaluate(awi, bcc):
    records = []
    for source, raw in (("AWI", awi), ("BCC", bcc)):
        validate_source(source, raw)
        for scenario in SSPS:
            for earlier, later in ((2030, 2050), (2050, 2080), (2030, 2080)):
                for model, weight in (("persistence", None),) + tuple(
                    (m, w) for m in MODELS for w in TAIL_WEIGHTS):
                    first = extract_bias(source, raw, scenario, earlier, model, weight)
                    last = extract_bias(source, raw, scenario, later, model, weight)
                    change_errors = [b-a for a,b in zip(first,last)]
                    if len(change_errors) != len(first):
                        raise ValueError("Unmatched model seeds")
                    records.append({
                        "gcm": source, "scenario":scenario, "first_year":earlier,
                        "last_year":later, "predictor":model,
                        "historical_train_tail_weight":None if weight is None else int(weight),
                        "n_pairs_each_year":24,
                        "individual_seed_change_errors_K":change_errors,
                        "mean_change_error_K":sum(change_errors)/len(change_errors),
                        "mean_abs_change_error_K":abs(sum(change_errors)/len(change_errors)),
                        "mean_absolute_seed_change_error_K":sum(map(abs,change_errors))/len(change_errors),
                        "bias_first_year_K":first, "bias_last_year_K":last,
                    })
    if len(records) != 2*2*3*7:
        raise ValueError("Incomplete interyear change-error factorial")
    return {
        "status":"v57_sparse_interyear_forecast_change_error_no_forced_warming_claim",
        "input_sources":{"AWI":"immutable_v48", "BCC":"immutable_v54"},
        "frozen_dates":{"years":list(YEARS),"days_per_month":[8,20]},
        "gcm_replication_count":2, "member_replication_count_per_gcm":1,
        "records":records,
        "scientific_scope":[
            "Bias difference on paired sparse calendar-day samples equals sampled predicted-minus-simulated temperature-change error",
            "The first and last years use matching dates but DIFFERENT weather events",
            "Not an estimate of forced global/regional warming trend or a Historical-to-SSP warming-amplitude preservation test",
            "No new model training, SSP target fitting, cherry-picking or selection",
            "Seed effects are repeated optimization fits, not independent climate ensemble members",
            "Do not interpret correlated scenario-year or gridcell observations as independent replications"
        ]
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--awi",required=True)
    p.add_argument("--bcc",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    dest=Path(args.output)
    if dest.exists():
        raise FileExistsError("Cannot overwrite frozen v5.7 diagnostic")
    report=evaluate(json.loads(Path(args.awi).read_text()),
                    json.loads(Path(args.bcc).read_text()))
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"records":len(report["records"])},indent=2))

if __name__=="__main__":
    main()
