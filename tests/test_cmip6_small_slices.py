"""Read-only small-slice CMIP6 audit must fail closed on broken store."""
import pytest
from scripts.audit_cmip6_small_slices import audit_manifest


def manifest():
    row={"source_id":"A","member_id":"r1i1p1f1","grid_label":"gn"}
    row.update({e:{"store":"gs://cmip6/"+e}
                for e in ("historical","ssp245","ssp585")})
    return {"status":"catalog_metadata_only_no_climate_fields_or_transfer_skill",
            "catalog_sha256":"abc","selected_candidates":[row]}


def ok(store):
    exp=store.rsplit("/",1)[-1]
    historical=exp=="historical"
    return {"status":"small_slice_cf_dates_and_kelvin_samples_verified",
            "store":store,
            "calendar":"365_day",
            "tas_units":"K",
            "start_year":1850 if historical else 2015,
            "end_year":2014 if historical else 2100,
            "spatial_grid_shape":[80,100]}


def test_three_scenarios_must_pass(monkeypatch):
    monkeypatch.setattr("scripts.audit_cmip6_small_slices.audit_store",ok)
    report=audit_manifest(manifest(),1)
    assert report["compatible_trios"]==1
    assert report["models"][0]["historical_ssp_daily_schema_small_slice_compatible"]


def test_missing_future_store_fails_closed(monkeypatch):
    def broken(store):
        if store.endswith("/ssp585"):
            return {"status":"failed_small_slice_or_cf_audit","error":"not accessible"}
        return ok(store)
    monkeypatch.setattr("scripts.audit_cmip6_small_slices.audit_store",broken)
    report=audit_manifest(manifest(),1)
    assert report["compatible_trios"]==0


def test_ssp_store_begins_2201_cannot_pose_as_2015_2100(monkeypatch):
    def late_extension(store):
        out=ok(store)
        if store.endswith("/ssp585"):
            out["start_year"]=2201
            out["end_year"]=2300
        return out
    monkeypatch.setattr("scripts.audit_cmip6_small_slices.audit_store",late_extension)
    result=audit_manifest(manifest(),1)
    assert result["compatible_trios"]==0
    assert not result["models"][0]["historical_ssp_daily_schema_small_slice_compatible"]


def test_future_ends_before_historical_is_invalid(monkeypatch):
    def bad_dates(store):
        out=ok(store)
        if store.endswith("/ssp245"):
            out["start_year"]=2013
        return out
    monkeypatch.setattr("scripts.audit_cmip6_small_slices.audit_store",bad_dates)
    assert audit_manifest(manifest(),1)["compatible_trios"]==0


def test_missing_catalog_match_rejected():
    with pytest.raises(ValueError, match="Require successful"):
        audit_manifest({"status":"not_verified"},1)
