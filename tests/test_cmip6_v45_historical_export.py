"""v4.5 historical-only AWI daily target cohort and training-only thresholds."""
from collections import Counter

import pytest

from scripts.export_cmip6_awi_v45_historical import (
    TRAIN_YEARS, VALID_YEARS, TRAIN_DAYS, VALID_DAYS,
    schedule, validate_plan, validate_manifest,
)


def catalog():
    entry={"source_id":"AWI-CM-1-1-MR",
           "member_id":"r1i1p1f1","grid_label":"gn",
           "historical":{"store":"gs://cmip6/CMIP6/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/day/tas/gn/version/"}}
    return {"status":"catalog_metadata_only_no_climate_fields_or_transfer_skill",
            "catalog_sha256":"checksum","selected_candidates":[entry]}


def test_train_val_dates_and_minimum_monthly_p95_support():
    plan=schedule()
    validate_plan(plan)
    assert len(plan)==1140
    assert len(TRAIN_YEARS)==25
    assert len(VALID_YEARS)==10
    assert TRAIN_DAYS==(6,16,26) and VALID_DAYS==(8,20)
    counts=Counter((fold,month) for fold,year,month,day in plan)
    assert all(counts["train",m]==75 for m in range(1,13))
    assert all(counts["validation",m]==20 for m in range(1,13))
    assert max(TRAIN_YEARS)<min(VALID_YEARS)


def test_no_ssp_in_training_cohort():
    plan=schedule()
    plan[0]=("ssp245",2030,1,6)
    with pytest.raises(ValueError,match="75 and validation 20"):
        validate_plan(plan)


def test_no_duplicate_or_modified_sample_schedule():
    plan=schedule()
    plan[2]=plan[1]
    with pytest.raises(ValueError,match="unique dates"):
        validate_plan(plan)


def test_historical_store_only():
    assert "/historical/" in validate_manifest(catalog())
    x=catalog()
    x["selected_candidates"][0]["historical"]["store"]="gs://cmip6/ScenarioMIP/model/ssp585/day/tas/"
    with pytest.raises(ValueError,match="Historical"):
        validate_manifest(x)


def test_missing_ensemble_rejected():
    x=catalog()
    x["selected_candidates"][0]["member_id"]="r4i1p1f1"
    with pytest.raises(ValueError,match="exactly-one"):
        validate_manifest(x)
