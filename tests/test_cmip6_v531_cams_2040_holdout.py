"""v5.31 CAMS 2040 independent temporal holdout gates."""
import inspect
import pytest
from scripts.evaluate_cmip6_v531_cams_2040_holdout import YEAR,STATUS,run,verify
from scripts.evaluate_cmip6_v520_cams_contiguous_summer import dates,evaluate
def test_year_is_fixed_but_original_2050_default_unchanged():
    assert YEAR==2040 and dates()[0]==(2050,6,1)
    assert dates(YEAR)[0]==(2040,6,1)
    assert inspect.signature(evaluate).parameters["year"].default==2050
def test_gate_before_any_2040_SSP_access():
    with pytest.raises(ValueError,match="18 original"):
        run({"checkpoints":{}},{})
def test_reject_non_2040_report():
    with pytest.raises(ValueError,match="provenance mismatch"):
        verify({"status":STATUS,"source_model":"CAMS-CSM1-0","year":2050,
                "ssp_fitted":False,"checkpoint_sha":{},"scenarios":{}})
def test_verify_2040_no_future_training_and_complete_original_arms():
    metrics={k:{"individual":[.5,.5,.5]} for k in ("rmse_K","event_day_recall","event_day_precision")}
    report={"status":STATUS,"year":YEAR,"source_model":"CAMS-CSM1-0",
        "ssp_fitted":False,"checkpoint_sha":{str(i):"hash" for i in range(18)},
        "scenarios":{}}
    for scenario in ("ssp245","ssp585"):
        report["scenarios"][scenario]={"n_pairs":91,
          "source":{"n_source_days":91,"n_bounded_regional_native_days_read":92,
                    "target_first":"2040-06-02","target_last":"2040-08-31"},
          "models":{m:{w:metrics for w in ("0","1","3")} for m in ("fno","unet")}}
    assert verify(report) is report
    report["scenarios"]["ssp245"]["source"]["target_first"]="2050-06-02"
    with pytest.raises(ValueError,match="2040 continuous"):verify(report)
