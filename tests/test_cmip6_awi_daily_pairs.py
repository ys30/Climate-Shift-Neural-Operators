"""AWI CMIP6 regional native daily pair guardrail tests."""
import numpy as np
import pytest
from scripts.audit_cmip6_awi_daily_pairs import contiguous_span, exact_daily_pair_indices, run


def test_coordinate_span_and_longitude_conversion():
    x=np.arange(0,360,0.9375)
    span=contiguous_span(x,235,260)
    assert x[span].min()>=235 and x[span].max()<=260
    assert len(x[span])>4


def test_disjoint_bad_coordinates_rejected():
    with pytest.raises(ValueError,match="consecutive"):
        contiguous_span(np.array([25,26,28,60,40,42,44]),25,50)


def test_exact_native_plus_one_day_on_360_day_calendar():
    import cftime
    units="days since 2000-01-01 12:00:00"
    dates=[cftime.datetime(2000,1,15,12,calendar="360_day"),
           cftime.datetime(2000,1,16,12,calendar="360_day"),
           cftime.datetime(2000,1,17,12,calendar="360_day")]
    nums=cftime.date2num(dates,units=units,calendar="360_day")
    a,b,x,y=exact_daily_pair_indices(nums,(2000,1,15),units,"360_day")
    assert (a,b)==(0,1)
    assert x[:10]=="2000-01-15" and y[:10]=="2000-01-16"


def test_missing_native_day_rejected():
    import cftime
    units="days since 2000-01-01 12:00:00"
    times=cftime.date2num([
        cftime.datetime(2000,1,15,12,calendar="360_day"),
        cftime.datetime(2000,1,17,12,calendar="360_day")],
        units=units,calendar="360_day")
    with pytest.raises(ValueError,match="exactly one native day"):
        exact_daily_pair_indices(times,(2000,1,15),units,"360_day")


def test_manifest_requires_awi_member_and_grid():
    with pytest.raises(ValueError,match="catalog"):
        run({"status":"wrong"})
    with pytest.raises(ValueError,match="Exactly one"):
        run({"status":"catalog_metadata_only_no_climate_fields_or_transfer_skill",
             "selected_candidates":[]})


def test_all_three_scenarios_reported(monkeypatch):
    def fake(store,experiment):
        return {"status":"real_native_cmip6_1day_regional_pairs_very_small_pilot",
                "calendar":"proleptic_gregorian",
                "regional_grid_shape":[27,27],
                "pairs":[{"source_date":"2010-01-15","target_date":"2010-01-16"}]*2}
    monkeypatch.setattr("scripts.audit_cmip6_awi_daily_pairs.audit_store",fake)
    model={"source_id":"AWI-CM-1-1-MR","member_id":"r1i1p1f1","grid_label":"gn"}
    model.update({x:{"store":"gs://cmip6/"+x} for x in ("historical","ssp245","ssp585")})
    report=run({"status":"catalog_metadata_only_no_climate_fields_or_transfer_skill",
                "catalog_sha256":"abc","selected_candidates":[model]})
    assert len(report["scenarios"])==3
    assert "not_training" in report["status"]
