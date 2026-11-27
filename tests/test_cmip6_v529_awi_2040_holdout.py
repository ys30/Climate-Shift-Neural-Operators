"""v5.29 2040 independent time-window validation gates."""
import pytest
from scripts.evaluate_cmip6_v529_awi_2040_contiguous_holdout import YEAR,STATUS,run,verify_output
from scripts.evaluate_cmip6_v520_cams_contiguous_summer import dates
from scripts.evaluate_cmip6_v524_awi_contiguous_summer import evaluate

def test_predeclared_2040_distinct_from_2050():
    assert YEAR==2040
    assert dates()[0]==(2050,6,1)
    assert "2040" in STATUS

def test_fails_without_all_18_Historical_frozen_models_before_any_SSP():
    with pytest.raises(ValueError,match="18 frozen checkpoints"):
        run({"models":{}},{})

def test_reject_source_report_masquerading_as_2040():
    with pytest.raises(ValueError,match="Missing complete"):
        verify_output({"status":STATUS,"year":2050,"ssp_fitted":False,
                       "checkpoint_sha":{},"scenarios":{}})

def test_2050_evaluator_unchanged_default():
    import inspect
    assert inspect.signature(evaluate).parameters["year"].default==2050

def test_2040_expected_native_days_checked():
    report={"status":STATUS,"year":YEAR,"ssp_fitted":False,
      "checkpoint_sha":{str(x):"sha" for x in range(18)},
      "scenarios":{}}
    for s in ("ssp245","ssp585"):
        source={"native_calendar":"proleptic_gregorian",
          "native_grid_shape":[26,27],
          "n_bounded_regional_native_days_read":92,
          "first_target":"2040-06-02 12:00:00",
          "last_target":"2040-08-31 12:00:00"}
        arm={"rmse_K":{"individual":[1.]*3},
             "event_day_recall":{"individual":[.6]*3},
             "event_day_precision":{"individual":[.7]*3}}
        report["scenarios"][s]={"n_pairs":91,"source":source,"models":{
          m:{w:arm for w in ("0","1","3")} for m in ("fno","unet")}}
    assert verify_output(report) is report
    report["scenarios"]["ssp585"]["source"]["first_target"]="2050-06-02"
    with pytest.raises(ValueError,match="Wrong 2040"):verify_output(report)
