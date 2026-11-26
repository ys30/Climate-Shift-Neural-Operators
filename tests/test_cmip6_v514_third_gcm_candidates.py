"""E1 v5.14 third GCM candidate exclusions and real date-coverage guards."""
import pytest
from scripts.audit_cmip6_v514_third_gcm_candidates import select_new_candidates,coverage_gates

def manifest():
    def row(g):
        return {"source_id":g,"member_id":"r1i1p1f1","grid_label":"gn",
                **{e:{"store":f"gs://cmip6/CMIP6/{e}/day/tas/gn/"}
                   for e in ("historical","ssp245","ssp585")}}
    return {"status":"catalog_metadata_only_no_climate_fields_or_transfer_skill",
            "selected_candidates":[row(g) for g in ("ACCESS-CM2","AWI-CM-1-1-MR",
                            "CAMS-CSM1-0","CESM2","CESM2-WACCM")]}

def source(exp):
    return {"status":"small_slice_cf_dates_and_kelvin_samples_verified",
            "start_year":1850 if exp=="historical" else 2015,
            "end_year":2014 if exp=="historical" else 2100,
            "tas_units":"K","calendar":"noleap",
            "spatial_grid_shape":[24,24],"spatial_dims":["lat","lon"]}

def test_select_two_new_distinct_candidates():
    s=select_new_candidates(manifest(),2)
    assert [r["source_id"] for r in s]==["CAMS-CSM1-0","CESM2"]

def test_compatibility_all_three_arms():
    ok,fail=coverage_gates({e:source(e) for e in ("historical","ssp245","ssp585")})
    assert ok and fail==[]

def test_extended_period_2201_is_ineligible():
    x={e:source(e) for e in ("historical","ssp245","ssp585")}
    x["ssp585"]["start_year"]=2201;x["ssp585"]["end_year"]=2300
    ok,fail=coverage_gates(x)
    assert not ok and "ssp585_missing_2015_to_2080" in fail

def test_native_calendar_mismatch_is_ineligible():
    x={e:source(e) for e in ("historical","ssp245","ssp585")}
    x["ssp585"]["calendar"]="360_day"
    ok,fail=coverage_gates(x)
    assert not ok and "calendar_mismatch" in fail

def test_failed_tiny_slice_is_not_valid():
    x={e:source(e) for e in ("historical","ssp245","ssp585")}
    x["ssp245"]["status"]="failed_small_slice_or_cf_audit"
    assert coverage_gates(x)==(False,["failed_tiny_source_read"])

def test_disallow_imaginary_sources():
    x=manifest()
    x["selected_candidates"][2]["ssp585"]["store"]="gs://not-the-real-catalog/"
    with pytest.raises(ValueError,match="Invalid source"):select_new_candidates(x,2)
