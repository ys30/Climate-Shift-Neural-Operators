"""Ensure confirmation reporting cannot silently drop poor-performing treatments."""
import copy
import pytest
from scripts.report_fresh_tail_confirmation import summarize


def fixture():
    models={}
    for model in ("fno","unet"):
        models[model]={
            "weighted_rmse":{"mean":3.0,"individual":[2.9,3.0,3.1]},
            "local_p95_recall":{"mean":.5,"individual":[.4,.5,.6]},
            "local_p95_precision_by_seed":[.65,.6,.55],
            "per_year":{
                str(y):{
                    "weighted_rmse":{"mean":3.0},
                    "local_p95_recall":{"mean":.5},
                    "hot_z_gt_2_rmse_by_seed":[2.5,2.6,2.7],
                } for y in (2015,2016,2017)
            },
        }
    return {
        "status":"fresh_heldout_confirmation_not_CMIP6_climate_shift",
        "candidate_tail_weights":[0,1,3],
        "heldout_years":[2015,2016,2017],
        "data_sha256":"same",
        "arms":{k:{"input_sha256":"same","models":copy.deepcopy(models)}
                for k in ("0","1","3")},
    }


def test_all_six_arms_reported_including_ties():
    result=summarize(fixture())
    assert len(result["arm_rows"])==6
    assert all(len(row["yearwise"])==3 for row in result["arm_rows"])
    assert result["paired_against_standard_mse"]["unet"]["1"]["weighted_rmse_K"]["mean"] == 0
    assert not result["paired_against_standard_mse"]["unet"]["1"]["weighted_rmse_K"]["all_seeds_favor_treatment"]


def test_no_dropping_failed_arm():
    data=fixture()
    del data["arms"]["3"]
    with pytest.raises(ValueError, match="one exact ERA5"):
        summarize(data)


def test_no_heldout_year_substitution():
    data=fixture()
    data["heldout_years"]=[2012,2013,2014]
    with pytest.raises(ValueError,match="Unexpected heldout"):
        summarize(data)
