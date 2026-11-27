"""v5.30 BCC 2040 noleap same frozen checkpoints temporal holdout."""
import pytest
from scripts.evaluate_cmip6_v530_bcc_2040_holdout import STATUS,YEAR,verify,evaluate_2040
from scripts.evaluate_cmip6_v523_bcc_contiguous_summer import evaluate
import inspect
def test_year_2040_distinct_from_original_2050_default():
    assert YEAR==2040 and "2040" in STATUS
    assert inspect.signature(evaluate).parameters["year"].default==2050
def test_original_18_required_before_any_future_read():
    with pytest.raises(ValueError,match="All 18"):evaluate_2040({"checkpoints":{}},{})
def test_reject_masquerading_2050_report():
    with pytest.raises(ValueError,match="provenance violation"):
        verify({"status":STATUS,"year":2050,"source_model":"BCC-CSM2-MR","ssp_fitted":False,"checkpoint_sha":{},"scenarios":{}})
def test_2040_native_noleap_contract_and_complete_loss_arms():
    arm={"rmse_K":{"individual":[1.5]*3},"event_day_recall":{"individual":[.6]*3},"event_day_precision":{"individual":[.7]*3}}
    a={"status":STATUS,"year":YEAR,"source_model":"BCC-CSM2-MR",
      "ssp_fitted":False,"checkpoint_sha":{str(i):"sha" for i in range(18)},
      "scenarios":{}}
    for s in ("ssp245","ssp585"):
        a["scenarios"][s]={"n_pairs":91,
          "source":{"native_calendar":"noleap","native_grid_shape":[23,23],
            "n_bounded_regional_native_days_read":92,
            "first_target":"2040-06-02 12:00:00","last_target":"2040-08-31 12:00:00"},
          "models":{m:{w:arm for w in ("0","1","3")} for m in ("fno","unet")}}
    assert verify(a) is a
    a["scenarios"]["ssp585"]["source"]["last_target"]="2050-08-31"
    with pytest.raises(ValueError,match="2040 day"):verify(a)
