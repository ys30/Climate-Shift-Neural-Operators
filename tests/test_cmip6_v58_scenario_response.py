"""E1 v5.8 SSP scenario difference bias identity and guards."""
import pytest
from scripts.evaluate_cmip6_v58_scenario_response import summarize
from tests.test_cmip6_v57_sampled_change import sample

def test_complete_scenario_response():
    a,b=sample()
    result=summarize(a,b)
    assert len(result["rows"])==42
    assert len([x for x in result["rows"] if x["predictor"]=="persistence"])==6
    assert all(x["mean_scenario_response_error_K"]==pytest.approx(0) for x in result["rows"])

def test_response_bias_identity():
    a,b=sample()
    p=b["scenario_results"]["ssp585"]["scores"]["2080"]["models"]["fno"]["0"]["weighted_bias_K"]
    p["individual"]=[x+0.25 for x in p["individual"]]
    p["mean"]+=0.25
    d=summarize(a,b)
    row=next(r for r in d["rows"] if r["gcm"]=="BCC" and r["year"]==2080
             and r["predictor"]=="fno" and r["historical_train_tail_weight"]==0)
    assert row["mean_scenario_response_error_K"]==pytest.approx(.25)
    assert row["mean_absolute_seed_response_error_K"]==pytest.approx(.25)

def test_reject_future_fitting_flag():
    a,b=sample()
    b["ssp_used_for_fitting_or_checkpoint_selection"]=True
    with pytest.raises(ValueError,match="gate failed"):
        summarize(a,b)

def test_reject_mismatched_pairs():
    a,b=sample()
    b["scenario_results"]["ssp245"]["scores"]["2050"]["n_pairs"]=20
    with pytest.raises(ValueError,match="24 dates"):
        summarize(a,b)

def test_reject_inconsistent_bias_mean():
    a,b=sample()
    a["scenarios"]["ssp585"]["groups"]["2030"]["models"]["unet"]["1"]["weighted_bias_K"]["mean"]=999
    with pytest.raises(ValueError,match="do not reproduce"):
        summarize(a,b)
