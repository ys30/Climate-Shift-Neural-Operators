"""E1 v5.10 derived physical scenario contrast identity and guards."""
import copy
import pytest
from scripts.join_cmip6_v510_source_response import build

def sample():
    source={"status":"v59_sparse_CMIP6_source_only_temperature_reference",
        "historical_reference":[2005,2014],"future_years":[2030,2050,2080],
        "future_source_days":[8,20],"rows":[]}
    error={"status":"v58_sparse_ssp585_minus_ssp245_scenario_response_error_no_forced_signal_attribution",
        "matched_dates":{"days_per_month":[8,20],"pairs_each_year":24},"rows":[]}
    for g in ("AWI","BCC"):
        for y in (2030,2050,2080):
            for s,offset in (("ssp245",1),("ssp585",3)):
                source["rows"].append(dict(gcm=g,scenario=s,year=y,
                    n_historical_pairs=240,n_future_pairs=24,
                    historical_regional_mean_K=285.0,
                    projected_sparse_regional_mean_K=285.0+offset,
                    sampled_change_vs_historical_K=float(offset)))
            for m in ("persistence","fno","unet"):
                for w in ((None,) if m=="persistence" else (0,1,3)):
                    seeds=[.5] if m=="persistence" else [.1,.2,.3]
                    error["rows"].append(dict(gcm=g,year=y,predictor=m,
                        historical_train_tail_weight=w,scenario_pair="ssp585_minus_ssp245",
                        date_pairs_per_scenario=24,mean_scenario_response_error_K=sum(seeds)/len(seeds),
                        per_seed_scenario_response_errors_K=seeds))
    return source,error

def test_full_join_and_exact_identity():
    r=build(*sample())
    assert len(r["rows"])==42
    x=next(z for z in r["rows"] if z["predictor"]=="fno")
    assert x["source_simulated_ssp585_minus_245_K"]==pytest.approx(2)
    assert x["forecast_ssp585_minus_245_K"]==pytest.approx(2.2)

def test_duplicate_source_rejected():
    s,e=sample();s["rows"].append(copy.deepcopy(s["rows"][0]))
    with pytest.raises(ValueError,match="Duplicate"):build(s,e)

def test_inconsistent_historical_reference_rejected():
    s,e=sample();s["rows"][0]["historical_regional_mean_K"]=284
    s["rows"][0]["sampled_change_vs_historical_K"]=2
    with pytest.raises(ValueError,match="same historical"):build(s,e)

def test_wrong_date_count_rejected():
    s,e=sample();e["rows"][0]["date_pairs_per_scenario"]=23
    with pytest.raises(ValueError,match="date count"):build(s,e)

def test_missing_weight_rejected():
    s,e=sample();e["rows"].pop()
    with pytest.raises(ValueError,match="Incomplete"):build(s,e)

def test_inconsistent_seeds_rejected():
    s,e=sample();e["rows"][1]["mean_scenario_response_error_K"]=0
    with pytest.raises(ValueError,match="not supported"):build(s,e)
