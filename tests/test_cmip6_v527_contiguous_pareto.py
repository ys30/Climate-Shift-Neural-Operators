"""E1 v5.27 Pareto-front regression and source-integrity tests."""
import csv
import pytest
from scripts.audit_cmip6_v527_contiguous_pareto import dominates,analyze,export
from tests.test_cmip6_v525_three_gcm_contiguous_events import fixture
from scripts.audit_cmip6_v525_three_gcm_contiguous_events import compare

def source():
    return compare(*fixture())

def test_dominance_requires_no_worse_all_three_and_strict_one():
    assert dominates((1.8,.7,.8),(2.,.6,.75))
    assert not dominates((1.8,.4,.8),(2.,.6,.75))
    assert not dominates((1.8,.7,.8),(1.8,.7,.8))

def test_all_three_GCM_42_candidates():
    r=analyze(source())
    assert r["candidate_rows"]==42 and len(r["scenarios"])==6
    assert all(x["n_candidates"]==7 for x in r["scenarios"])
    assert r["independent_GCM_count"]==3
    assert r["no_future_refit"] is True

def test_persistence_always_is_seventh_candidate():
    r=analyze(source())
    assert sum(x["model"]=="persistence" for x in r["candidate_dominance"])==6

def test_null_precision_is_not_misrepresented_as_zero():
    d=source()
    d["neural_arms"][0]["model_event_day_precision"]=None
    with pytest.raises(ValueError,match="Invalid numerical"):analyze(d)

def test_reject_mismatched_persistence_between_arms():
    d=source()
    d["neural_arms"][1]["persistence_rmse_K"]+=.2
    with pytest.raises(ValueError,match="Persistence not held fixed"):analyze(d)

def test_reject_changed_checkpoint_inventory():
    d=source();d["frozen_checkpoint_count"]=53
    with pytest.raises(ValueError,match="complete"):analyze(d)

def test_write_all_results(tmp_path):
    r=analyze(source());p=tmp_path/"pareto"
    export(r,p)
    with (p/"v527_42_candidate_metrics.csv").open() as f:
        assert len(list(csv.DictReader(f)))==42
    assert "not an inferential sample" in (p/"v527_three_GCM_multiobjective_results.md").read_text()
    with pytest.raises(FileExistsError):export(r,p)
