"""Frozen BCC native noleap SSP test invariants, not benchmark metrics."""
import pytest
from scripts.evaluate_cmip6_bcc_v54_frozen_ssp import (
    YEARS,DAYS,planned_days,aggregate_seed_scores
)


def test_ssp_schedule_frozen_before_training_results():
    dates=planned_days()
    assert len(dates)==72 and len(set(dates))==72
    assert {x[0] for x in dates}==set(YEARS)=={2030,2050,2080}
    assert {x[2] for x in dates}==set(DAYS)=={8,20}


def test_full_three_seed_summary_or_fail():
    row={
        "weighted":{"rmse":2.1,"mae":1.5,"bias":-.15},
        "monthly_hot":{"hot_recall":.6,"hot_precision":.7,
                       "predicted_hot_fraction":.15},
    }
    score=aggregate_seed_scores([row,row,row])
    assert score["weighted_rmse_K"]["mean"]==2.1
    assert score["local_p95_precision"]["mean"] == pytest.approx(.7)
    with pytest.raises(ValueError,match="three frozen"):
        aggregate_seed_scores([row,row])


def test_missing_future_hot_metric_rejected():
    row={
        "weighted":{"rmse":2.1,"mae":1.5,"bias":-.15},
        "monthly_hot":{"hot_recall":.6,"hot_precision":None,
                       "predicted_hot_fraction":.15},
    }
    with pytest.raises(ValueError,match="Invalid BCC future"):
        aggregate_seed_scores([row,row,row])
