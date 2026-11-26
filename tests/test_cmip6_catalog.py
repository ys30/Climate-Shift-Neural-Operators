"""CMIP6 strict metadata-only pairing, no manufactured SSP matches."""
import copy
import pytest
from scripts.audit_cmip6_catalog import select_pairs


def row(experiment, model="Model-A", member="r1i1p1f1", grid="gn",
        table="day", variable="tas", activity=None, version="20240101"):
    if activity is None:
        activity = "CMIP" if experiment=="historical" else "ScenarioMIP"
    return dict(
        activity_id=activity, source_id=model, experiment_id=experiment,
        member_id=member, table_id=table, variable_id=variable,
        grid_label=grid, version=version,
        zstore=f"gs://cmip6/{activity}/Centre/{model}/{experiment}/{member}/{table}/{variable}/{grid}/",
    )


def test_exact_model_member_grid_and_daily_variable():
    rows = [row("historical"), row("ssp245"), row("ssp585"),
            row("ssp585",model="Model-B"), row("historical",variable="pr"),
            row("ssp585",table="Amon")]
    report=select_pairs(rows,"sha",max_models=3)
    assert report["status"]=="catalog_metadata_only_no_climate_fields_or_transfer_skill"
    assert report["matching_model_members"]==1
    assert report["selected_candidates"][0]["source_id"]=="Model-A"
    assert set(report["selected_candidates"][0]) == {
        "source_id","member_id","grid_label","historical","ssp245","ssp585"
    }


def test_do_not_mix_member_or_grid_to_fake_matching():
    rows=[row("historical"),row("ssp245",member="r2i1p1f1"),
          row("ssp585",grid="gr")]
    assert select_pairs(rows,"sha")["selected_candidates"]==[]


def test_version_duplicates_are_one_ensemble_and_latest_is_selected():
    rows=[row("historical",version="20230101"),row("historical",version="20250101"),
          row("ssp245"),row("ssp585")]
    report=select_pairs(rows,"sha")
    assert report["matching_model_members"]==1
    assert report["selected_candidates"][0]["historical"]["duplicate_catalog_rows"]==2
    assert report["selected_candidates"][0]["historical"]["version"]=="20250101"


def test_no_monthly_fallback_even_if_three_experiments_available():
    rows=[row(z,table="Amon") for z in ("historical","ssp245","ssp585")]
    assert select_pairs(rows,"sha")["selected_candidates"] == []


def test_only_correct_activity_and_google_cmip6_prefix():
    rows=[row(z) for z in ("historical","ssp245","ssp585")]
    rows[1]["activity_id"]="CMIP"
    assert not select_pairs(rows,"sha")["selected_candidates"]
    rows[1]["activity_id"]="ScenarioMIP"
    rows[0]["zstore"]="gs://unrelated/"
    assert not select_pairs(rows,"sha")["selected_candidates"]


def test_bounded_model_count():
    rows=[row(e) for e in ("historical","ssp245","ssp585")]
    with pytest.raises(ValueError,match="1–20"):
        select_pairs(rows,"sha",max_models=0)
