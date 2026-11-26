"""E1 v5.15 CAMS native-grid preflight checks, no training."""
import copy
import pytest
from scripts.audit_cmip6_v515_cams_regional_pairs import select_verified,validate_regional,EXPS

def fixtures():
    source={"source_id":"CAMS-CSM1-0","member_id":"r2i1p1f1","grid_label":"gn",
            **{e:{"store":f"gs://cmip6/CMIP6/{e}/day/tas/gn/"} for e in EXPS}}
    metadata={"status":"catalog_metadata_only_no_climate_fields_or_transfer_skill",
              "catalog_sha256":"audit-sha","selected_candidates":[source]}
    prior={"status":"v514_candidate_third_GCM_native_daily_source_small_slice_eligibility",
           "catalog_sha256":"audit-sha","models":[{
               "source_id":"CAMS-CSM1-0","member_id":"r2i1p1f1","grid_label":"gn",
               "passed_small_slice_time_and_geometry_gate":True,
               "metadata_provenance":{e:copy.deepcopy(source[e]) for e in EXPS}}]}
    return metadata,prior

def native():
    return {e:{"status":"actual_bcc_native_one_day_pair_sanity_only",
        "grid_shape":[12,13],"latitude_sha256":"a","longitude_sha256":"b",
        "calendar":"noleap",
        "pairs":[{"n_cells":156,"finite_values":True,
                  "persistence_weighted_rmse_K_sanity_only":2.0} for i in range(2)]}
        for e in EXPS}

def test_select_exact_verified_source():
    m,p=fixtures()
    assert select_verified(m,p)["member_id"]=="r2i1p1f1"

def test_reject_changed_store():
    m,p=fixtures()
    m["selected_candidates"][0]["ssp585"]["store"]="gs://cmip6/changed"
    with pytest.raises(ValueError,match="changed"):select_verified(m,p)

def test_reject_prior_not_passed():
    m,p=fixtures()
    p["models"][0]["passed_small_slice_time_and_geometry_gate"]=False
    with pytest.raises(ValueError,match="did not pass"):select_verified(m,p)

def test_valid_regional_shape():
    assert validate_regional(native())==(12,13)

def test_reject_different_native_grid():
    x=native();x["ssp585"]["longitude_sha256"]="different"
    with pytest.raises(ValueError,match="geometry"):validate_regional(x)

def test_reject_unfinished_pair_count():
    x=native();x["ssp245"]["pairs"].pop()
    with pytest.raises(ValueError,match="exactly two"):validate_regional(x)
