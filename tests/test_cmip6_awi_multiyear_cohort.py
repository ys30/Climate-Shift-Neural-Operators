"""Lock CMIP6 v4.4 independent yearly folds, daily source/target schedule and ensemble."""
import pytest
from scripts.audit_cmip6_awi_multiyear_cohort import (
    expected_schedule, validate_schedule, verify_member, run)


def fixture_manifest():
    model={"source_id":"AWI-CM-1-1-MR","member_id":"r1i1p1f1","grid_label":"gn"}
    model.update({e:{"store":"gs://cmip6/"+e}
                  for e in ("historical","ssp245","ssp585")})
    return {"status":"catalog_metadata_only_no_climate_fields_or_transfer_skill",
            "catalog_sha256":"test","selected_candidates":[model]}


def test_locked_sample_counts_and_fold_identity():
    plan=expected_schedule()
    validate_schedule(plan)
    assert len(plan["historical"])==168
    assert len(plan["ssp245"])==len(plan["ssp585"])==72
    assert sum(map(len,plan.values()))==312
    train={year for fold,year,month,day in plan["historical"] if fold=="train"}
    valid={year for fold,year,month,day in plan["historical"] if fold=="validation"}
    assert train=={1980,1990,2000,2004}
    assert valid=={2005,2010,2014}
    assert max(train)<min(valid)
    assert {year for fold,year,_,_ in plan["ssp245"]}=={2030,2050,2080}
    assert all(day in (8,20) for rows in plan.values() for _,_,_,day in rows)


def test_all_three_experiments_and_future_only_test():
    plan=expected_schedule()
    plan["ssp245"][0]=("train",2030,1,8)
    with pytest.raises(ValueError,match="leaked"):
        validate_schedule(plan)


def test_duplicates_rejected():
    plan=expected_schedule()
    plan["historical"][1]=plan["historical"][0]
    with pytest.raises(ValueError,match="Duplicate"):
        validate_schedule(plan)


def test_ensemble_identity():
    assert verify_member(fixture_manifest())["member_id"]=="r1i1p1f1"
    bad=fixture_manifest()
    bad["selected_candidates"][0]["member_id"]="r2i1p1f1"
    with pytest.raises(ValueError,match="exactly one"):
        verify_member(bad)


def test_full_run_fails_when_grid_mismatches(monkeypatch):
    def mock(store,scheduled):
        return {"status":"real_multiple_year_sparse_native_daily_pairs_checked",
                "calendar":"gregorian",
                "regional_grid_shape":[26,27] if store.endswith("historical") else [25,27],
                "n_pairs":len(scheduled),"per_fold_counts":{
                    fold:sum(x[0]==fold for x in scheduled)
                    for fold in {x[0] for x in scheduled}}}
    monkeypatch.setattr("scripts.audit_cmip6_awi_multiyear_cohort.audit_one_store",mock)
    with pytest.raises(ValueError,match="grid"):
        run(fixture_manifest())


def test_all_arms_audited_without_future_training(monkeypatch):
    def mock(store,scheduled):
        return {"status":"real_multiple_year_sparse_native_daily_pairs_checked",
                "calendar":"gregorian","regional_grid_shape":[26,27],
                "n_pairs":len(scheduled),"per_fold_counts":{
                    fold:sum(x[0]==fold for x in scheduled)
                    for fold in {x[0] for x in scheduled}}}
    monkeypatch.setattr("scripts.audit_cmip6_awi_multiyear_cohort.audit_one_store",mock)
    out=run(fixture_manifest())
    assert out["n_pairs_total"]==312
    assert out["status"]=="v44_sparse_multiyear_data_audit_no_neural_forecasting"
