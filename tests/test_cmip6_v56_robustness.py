"""Unit and corruption tests for immutable v5.6 audit."""
import copy
import pytest
from scripts.audit_cmip6_v56_robustness import audit

def example():
    rows=[]
    for g in ("AWI","BCC"):
        for s in ("ssp245","ssp585"):
            for y in ("all","2030","2050","2080"):
                for m in ("fno","unet"):
                    for w in (1,2,4):
                        rmse=2.0+(w-1)*.01
                        recall=.5+(w-1)*.03
                        rows.append(dict(source_gcm=g,ssp=s,scope=y,model=m,
                            training_hot_multiplier=w,n_source_target_pairs=72 if y=="all" else 24,
                            weighted_rmse_K=rmse,persistence_weighted_rmse_K=2.5,
                            rmse_improvement_fraction_vs_own_persistence=1-rmse/2.5,
                            local_p95_recall=recall,persistence_local_p95_recall=.7,
                            recall_delta_vs_own_persistence=recall-.7,local_p95_precision=.6))
    return {"status":"paired_within_GCM_projected_temperature_forecast_contrasts_no_pooled_significance","rows":rows}

def test_complete_and_descriptive_only():
    result=audit(example())
    assert result["direction_counts"]["year_cells"]==12
    assert result["direction_counts"]["fno1_rmse_better_years"]==12
    assert result["independent_gcm_count"]==2
    assert "p_value" not in str(result)

@pytest.mark.parametrize("change,error",[
    ("drop","Incomplete"),
    ("duplicate","Duplicated"),
    ("bias","Inconsistent native"),
    ("nan","non-finite"),
    ("count","date-count")
])
def test_corruptions_rejected(change,error):
    x=example()
    if change=="drop":x["rows"].pop()
    elif change=="duplicate":x["rows"].append(copy.deepcopy(x["rows"][0]))
    elif change=="bias":x["rows"][0]["rmse_improvement_fraction_vs_own_persistence"]=.1
    elif change=="nan":x["rows"][0]["weighted_rmse_K"]=float("nan")
    else:x["rows"][0]["n_source_target_pairs"]=13
    with pytest.raises(ValueError,match=error):
        audit(x)
