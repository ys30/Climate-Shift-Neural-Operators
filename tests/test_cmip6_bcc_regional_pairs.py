"""Second-model BCC native-grid and scenario isolation tests."""
import pytest
from scripts.audit_cmip6_bcc_regional_pairs import DATES, get_model, audit


def manifest():
    row={"source_id":"BCC-CSM2-MR","member_id":"r1i1p1f1","grid_label":"gn"}
    row.update({s:{"store":f"gs://cmip6/CMIP6/{'CMIP' if s=='historical' else 'ScenarioMIP'}/BCC/BCC-CSM2-MR/{s}/r1i1p1f1/day/tas/gn/v1/"}
                for s in DATES})
    return {"status":"catalog_metadata_only_no_climate_fields_or_transfer_skill",
            "catalog_sha256":"abc","selected_candidates":[row]}


def test_fixed_two_dates_each_scenario():
    assert set(DATES)=={"historical","ssp245","ssp585"}
    assert all(len(d)==2 for d in DATES.values())
    assert {year for d in (DATES["ssp245"],DATES["ssp585"]) for year,_,_ in d}=={2030,2050}


def test_reject_wrong_member_or_ssp_path():
    m=manifest()
    assert get_model(m)["source_id"]=="BCC-CSM2-MR"
    m["selected_candidates"][0]["member_id"]="r2i1p1f1"
    with pytest.raises(ValueError,match="exactly one"):
        get_model(m)
    m=manifest()
    m["selected_candidates"][0]["ssp585"]["store"]=m["selected_candidates"][0]["historical"]["store"]
    with pytest.raises(ValueError,match="frozen BCC"):
        get_model(m)


def test_only_same_coordinates_accepted(monkeypatch):
    def fake(store, dates):
        return {"status":"actual_bcc_native_one_day_pair_sanity_only",
                "grid_shape":[25,25],"latitude_sha256":"LAT","longitude_sha256":"LON",
                "calendar":"proleptic_gregorian",
                "pairs":[{"finite_values":True}]*2}
    monkeypatch.setattr("scripts.audit_cmip6_bcc_regional_pairs.sample_one",fake)
    report=audit(manifest())
    assert set(report["scenario_diagnostics"])==set(DATES)
    assert "no_neural_training" in report["status"]


def test_same_shape_but_different_grid_coordinates_is_rejected(monkeypatch):
    def fake(store, dates):
        return {"status":"actual_bcc_native_one_day_pair_sanity_only",
                "grid_shape":[25,25],
                "latitude_sha256":"LAT" if "/historical/" in store else "SHIFT",
                "longitude_sha256":"LON","calendar":"proleptic_gregorian",
                "pairs":[{"finite_values":True}]*2}
    monkeypatch.setattr("scripts.audit_cmip6_bcc_regional_pairs.sample_one",fake)
    with pytest.raises(ValueError,match="coordinate values"):
        audit(manifest())
