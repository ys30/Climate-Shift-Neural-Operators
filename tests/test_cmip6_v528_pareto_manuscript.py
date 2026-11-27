"""E1 v5.28: plot-data source validation and immutable Pareto front checks."""
import csv
import pytest
from scripts.render_cmip6_v528_pareto_manuscript import verified,draw
from tests.test_cmip6_v527_contiguous_pareto import source
from scripts.audit_cmip6_v527_contiguous_pareto import analyze

def report():
    return analyze(source())

def test_validate_all_42_original_options():
    rows=verified(report())
    assert len(rows)==42 and sum(r["model"]=="persistence" for r in rows)==6

def test_reject_fabricated_extra_ssp_or_duplicate():
    d=report();d["candidate_dominance"][0]["gcm"]="CAMS"
    with pytest.raises(ValueError,match="Duplicate|Missing expected"):verified(d)

def test_reject_tampered_Pareto_labels():
    d=report();d["candidate_dominance"][0]["pareto_nondominated"]=not d["candidate_dominance"][0]["pareto_nondominated"]
    with pytest.raises(ValueError,match="inconsistent"):verified(d)

def test_reject_future_fitting():
    d=report();d["no_future_refit"]=False
    with pytest.raises(ValueError,match="complete v5.27"):verified(d)

def test_real_plot_schema_and_results(tmp_path):
    pytest.importorskip("matplotlib")
    data=verified(report());out=tmp_path/"paper"
    draw(data,out)
    assert len(list(out.glob("*.png")))==2
    assert all(x.stat().st_size>6000 for x in out.glob("*.png"))
    with (out/"table_v528_original_42_candidate_metrics.csv").open() as f:
        assert len(list(csv.DictReader(f)))==42
    assert "should not be interpreted as a pre-registered model-selection result" in (out/"E1_v528_manuscript_Pareto_results.md").read_text()
    with pytest.raises(FileExistsError):draw(data,out)
