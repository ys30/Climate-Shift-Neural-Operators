"""No pseudo-replication: contrast AWI and BCC within native-grid SSP baselines."""
import copy
import pytest
from scripts.compare_cmip6_v55_awi_bcc import compare


def data():
    m={
        name:{w:{"weighted_rmse_K":{"mean":2.0},
                 "local_p95_recall":{"mean":.5},
                 "local_p95_precision":{"mean":.6}}
              for w in ("0","1","3")}
        for name in ("fno","unet")
    }
    awi={"status":"first_frozen_historical_to_sparse_SSP_forecast_evaluation",
         "ssp_used_for_fitting_or_selection":False,
         "historical_npz_sha256":"ac0a354032836c5de5463b122d26fad2270fd5ba2ebb48f43074105c8ce7b914",
         "future_years":[2030,2050,2080],"future_month_days":[8,20],
         "checkpoint_sha256":{str(i):"sha" for i in range(18)},
         "scenarios":{}}
    bcc={"status":"first_independent_GCM_BCC_Historical_to_sparse_SSP_frozen_forecast",
         "ssp_used_for_fitting_or_checkpoint_selection":False,
         "historical_source_sha256":"a23260e8f14a23fb57483038bcd6ca03ba74c7944a48ba29d62bcc5bd354f452",
         "future_years":[2030,2050,2080],"future_month_days":[8,20],
         "checkpoint_sha":{str(i):"sha" for i in range(18)},
         "scenario_results":{}}
    for scenario in ("ssp245","ssp585"):
        aa={};bb={}
        for scope in ("all","2030","2050","2080"):
            n=72 if scope=="all" else 24
            aa[scope]={"n_pairs":n,
                       "persistence":{"weighted":{"rmse":2.5},
                                      "local_p95":{"hot_recall":.7}},
                       "models":copy.deepcopy(m)}
            bb[scope]={"n_pairs":n,
                       "persistence":{"weighted":{"rmse":3.0},
                                      "monthly_hot":{"hot_recall":.65}},
                       "models":copy.deepcopy(m)}
        awi["scenarios"][scenario]={"groups":aa}
        bcc["scenario_results"][scenario]={"scores":bb}
    return awi,bcc


def test_full_cross_gcm_contrast_no_pooling():
    a,b=data()
    result=compare(a,b)
    assert len(result["rows"])==96
    subset=[r for r in result["rows"] if r["scope"]=="all" and r["model"]=="fno" and r["training_tail_weight"]==0]
    assert len(subset)==4
    assert {r["source_gcm"] for r in subset}=={"AWI","BCC"}
    assert {round(r["rmse_improvement_fraction_vs_own_persistence"],6) for r in subset}=={.2,round(1-2/3,6)}


def test_same_wrong_future_schedule_is_not_accepted():
    a,b=data()
    a["future_years"]=b["future_years"]=[2030,2040,2080]
    with pytest.raises(ValueError,match="year plan"):
        compare(a,b)


def test_no_missing_scenarios_or_checkpoint_data():
    a,b=data()
    b["checkpoint_sha"].pop("0")
    with pytest.raises(ValueError,match="18 historical"):
        compare(a,b)
