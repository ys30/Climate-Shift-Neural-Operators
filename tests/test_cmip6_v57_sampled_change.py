"""Synthetic schema tests: never confuse year-to-year bias with trend truth."""
import copy
import pytest
from scripts.evaluate_cmip6_v57_sampled_change import evaluate, AWI_SHA, BCC_SHA

def sample():
    def grp(year,source):
        bias=(year-2030)/50.0
        score={"weighted_bias_K":{"individual":[bias+.01,bias,bias-.01],"mean":bias}}
        models={m:{w:copy.deepcopy(score) for w in ("0","1","3")} for m in ("fno","unet")}
        return {"n_pairs":24,"persistence":{"weighted":{"bias":bias+.2}},
                "models":models}
    a={"status":"first_frozen_historical_to_sparse_SSP_forecast_evaluation",
       "historical_npz_sha256":AWI_SHA,"ssp_used_for_fitting_or_selection":False,
       "checkpoint_sha256":{str(i):"sha" for i in range(18)},
       "future_years":[2030,2050,2080],"future_month_days":[8,20],
       "scenarios":{s:{"groups":{str(y):grp(y,"AWI") for y in (2030,2050,2080)}} for s in ("ssp245","ssp585")}}
    b={"status":"first_independent_GCM_BCC_Historical_to_sparse_SSP_frozen_forecast",
       "historical_source_sha256":BCC_SHA,
       "ssp_used_for_fitting_or_checkpoint_selection":False,
       "checkpoint_sha":{str(i):"sha" for i in range(18)},
       "native_calendar":"noleap","future_years":[2030,2050,2080],
       "future_month_days":[8,20],
       "scenario_results":{s:{"scores":{str(y):grp(y,"BCC") for y in (2030,2050,2080)}} for s in ("ssp245","ssp585")}}
    return a,b

def test_exact_factorial_and_bias_difference():
    r=evaluate(*sample())
    assert len(r["records"])==84
    x=next(z for z in r["records"] if z["gcm"]=="AWI" and
           z["scenario"]=="ssp245" and z["first_year"]==2030 and
           z["last_year"]==2080 and z["predictor"]=="fno" and
           z["historical_train_tail_weight"]==0)
    assert x["mean_change_error_K"]==pytest.approx(1)
    assert r["gcm_replication_count"]==2

def test_reject_changed_schedule():
    a,b=sample()
    b["future_month_days"]=[1,15]
    with pytest.raises(ValueError,match="sampling schedule"):
        evaluate(a,b)

def test_reject_changed_checkpoint_count():
    a,b=sample()
    a["checkpoint_sha256"].pop("0")
    with pytest.raises(ValueError,match="gate failed"):
        evaluate(a,b)

def test_reject_missing_seed():
    a,b=sample()
    x=a["scenarios"]["ssp245"]["groups"]["2030"]["models"]["fno"]["0"]["weighted_bias_K"]
    x["individual"].pop()
    with pytest.raises(ValueError,match="three frozen seeds"):
        evaluate(a,b)

def test_reject_nonfinite():
    a,b=sample()
    b["scenario_results"]["ssp585"]["scores"]["2080"]["persistence"]["weighted"]["bias"]=float("nan")
    with pytest.raises(ValueError,match="Nonfinite"):
        evaluate(a,b)

def test_reject_missing_sample_date():
    a,b=sample()
    a["scenarios"]["ssp245"]["groups"]["2050"]["n_pairs"]=23
    with pytest.raises(ValueError,match="24 dates"):
        evaluate(a,b)
