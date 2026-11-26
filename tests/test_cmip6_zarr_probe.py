"""Do not silently treat missing CMIP6 public Zarr stores as verified."""
from scripts.probe_cmip6_zarr import audit, inspect_store


def _manifest():
    d={"status":"catalog_metadata_only_no_climate_fields_or_transfer_skill",
       "variables":["tas"],"table_id":"day","catalog_sha256":"abc",
       "selected_candidates":[]}
    entry={"source_id":"A","member_id":"r1","grid_label":"gn"}
    entry.update({e:{"store":"gs://cmip6/model/"+e}
                  for e in ("historical","ssp245","ssp585")})
    d["selected_candidates"]=[entry]
    return d


def test_probe_rejects_missing_ensemble_without_claiming_success(monkeypatch):
    monkeypatch.setattr("scripts.probe_cmip6_zarr.inspect_store",
        lambda store:{"status":"unverified_store_or_schema",
                      "store":store,"error":"not found"})
    out=audit(_manifest(),1)
    assert out["complete_trios"]==0
    assert not out["models"][0]["matched_trio_accessible_and_compatible"]


def test_probe_records_actual_compatible_trio(monkeypatch):
    monkeypatch.setattr("scripts.probe_cmip6_zarr.inspect_store",
        lambda store:{"status":"opened_and_schema_checked_metadata_only",
                      "store":store,"tas_units":"K","time_calendar":"365_day",
                      "tas_shape":[30000,120,240]})
    out=audit(_manifest(),1)
    assert out["complete_trios"]==1


def test_day_monthly_metadata_mismatch_not_accepted():
    d=_manifest()
    d["table_id"]="Amon"
    try:
        audit(d,1)
    except ValueError as exc:
        assert "verified daily tas" in str(exc)
    else:
        raise AssertionError("Monthly tas cannot pass daily-tas gate")


def test_wrong_external_bucket_rejected():
    try:
        inspect_store("gs://fake/model")
    except ValueError as exc:
        assert "gs://cmip6/" in str(exc)
    else:
        raise AssertionError("Unexpected bucket should fail")
