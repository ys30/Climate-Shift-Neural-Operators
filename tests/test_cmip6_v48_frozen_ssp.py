"""Freeze all SSP forecast arms, historical provenance, source identity and no refit."""
import pytest

from scripts.evaluate_cmip6_v48_frozen_ssp import (
    YEARS, DAYS, plan, manifest_entry, summarize_seed_scores
)


def test_prospective_sparse_ssp_dates_unchanged():
    p=plan()
    assert len(p)==72 and len(set(p))==72
    assert {t[0] for t in p}=={2030,2050,2080}
    assert {t[2] for t in p}=={8,20}
    assert [sum(y==year for y,_,_ in p) for year in YEARS]==[24,24,24]
    assert DAYS==(8,20)


def test_scenario_source_requires_exact_awi_model_member_grid():
    entry={"source_id":"AWI-CM-1-1-MR","member_id":"r1i1p1f1","grid_label":"gn",
           "ssp245":{"store":"gs://cmip6/ScenarioMIP/AWI/AWI-CM-1-1-MR/ssp245/r1i1p1f1/day/tas/gn/version"},
           "ssp585":{"store":"gs://cmip6/ScenarioMIP/AWI/AWI-CM-1-1-MR/ssp585/r1i1p1f1/day/tas/gn/version"}}
    manifest={"status":"catalog_metadata_only_no_climate_fields_or_transfer_skill",
              "selected_candidates":[entry]}
    assert manifest_entry(manifest)["grid_label"]=="gn"
    entry["ssp585"]["store"]="gs://cmip6/CMIP/other/historical/r1/day/tas/gn/"
    with pytest.raises(ValueError,match="future source"):
        manifest_entry(manifest)


def test_no_missing_seed_metrics_or_partial_report():
    baseline={"weighted":{"rmse":2.3}}
    m={"weighted":{"rmse":2.0,"mae":1.3,"bias":-0.1},
       "local_p95":{"hot_recall":0.5,"hot_precision":0.7,
                    "predicted_hot_fraction":0.06}}
    report=summarize_seed_scores([m,m,m],baseline)
    assert report["weighted_rmse_K"]["individual"]==[2.0]*3
    with pytest.raises(ValueError,match="No incomplete"):
        summarize_seed_scores([m,m],baseline)
    m["local_p95"]["hot_recall"]=None
    with pytest.raises(ValueError,match="nonfinite"):
        summarize_seed_scores([m,m,m],baseline)
