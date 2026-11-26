"""BCC v5.2 noleap model train/metric isolation and parameter guards."""
import numpy as np
import pytest
from scripts.fit_cmip6_bcc_v52_historical import (
    monthly_local_hot, validate_historical, fit
)


def test_month_threshold_uses_native_months_not_gregorian_dates():
    p95=np.full((12,2,3),290.0,dtype=np.float32)
    p95[1]=270.0
    lat=np.array([35.,40.])
    pred=np.full((2,1,2,3),285.,dtype=np.float32)
    truth=np.full((2,1,2,3),280.,dtype=np.float32)
    x=monthly_local_hot(pred,truth,np.array([1,2]),p95,lat)
    assert x["hot_recall"]==1.0
    assert x["hot_precision"]==1.0
    assert abs(x["reference_hot_fraction"]-0.5)<1e-6


def test_wrong_hash_cannot_train(tmp_path):
    p=tmp_path/"fake.npz"
    p.write_bytes(b"fake")
    with pytest.raises(ValueError,match="hash mismatch"):
        validate_historical(p,"0"*64)


def test_no_unapproved_training_configuration(tmp_path):
    with pytest.raises(ValueError,match="Unfrozen"):
        fit("irrelevant.npz","0"*64,str(tmp_path/"r"),epochs=101)
    with pytest.raises(ValueError,match="Unfrozen"):
        fit("irrelevant.npz","0"*64,str(tmp_path/"r"),tail_weight=2)
    with pytest.raises(ValueError,match="Unfrozen"):
        fit("irrelevant.npz","0"*64,str(tmp_path/"r"),seed=999)
