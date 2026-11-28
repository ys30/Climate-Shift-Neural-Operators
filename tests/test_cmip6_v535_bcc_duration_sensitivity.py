"""v5.35 BCC 2/3/5-day source-gated sensitivity tests."""
import numpy as np
import pytest
from scripts.evaluate_cmip6_v535_bcc_duration_sensitivity import STATUS,gate,analyze
from scripts.evaluate_cmip6_v534_awi_duration_sensitivity import DURATIONS,YEARS,SSPS,event

def test_same_event_rule_as_AWI_with_native_BCC_shape():
    truth=np.full((91,1,23,23),295.)
    p95=np.full((12,23,23),300.)
    truth[15:18,0,2,4]=305.
    result=event(truth,truth,np.full(91,6,dtype=int),p95,np.linspace(25,50,23),3)
    assert result["event_day_recall"]==pytest.approx(1)
    assert result["event_day_precision"]==pytest.approx(1)

def test_source_gate_rejects_checkpoint_change_before_any_SSP_read():
    frozen={"checkpoint_sha_by_model":{"a":"good"}}
    report={"year":2040,"source_model":"BCC-CSM2-MR","ssp_fitted":False,
      "checkpoint_sha":{"a":"bad"},"scenarios":{"ssp245":{},"ssp585":{}}}
    with pytest.raises(ValueError,match="checkpoint SHA"):gate(report,2040,frozen)

def test_reject_incomplete_original_historical_checkpoints():
    with pytest.raises(ValueError,match="all 18"):
        analyze({"checkpoints":{},"checkpoint_sha_by_model":{}},{},{},{})

def test_duration_year_and_scenario_declarations():
    assert "BCC" in STATUS and YEARS==(2040,2050)
    assert SSPS==("ssp245","ssp585") and DURATIONS==(2,3,5)
