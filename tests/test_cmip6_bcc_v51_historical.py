"""BCC v5.1 noleap source/target and monthly-P95 isolation safeguards."""
import numpy as np
import pytest
from scripts.export_cmip6_bcc_v51_historical import (
    schedule, validate_schedule, monthly_training_reference, native_noleap_pair
)


def test_fixed_25_year_train_10_year_val_noleap_schedule():
    p=schedule()
    validate_schedule(p)
    assert len(p)==1140 and len(set(p))==1140
    assert sum(f=="train" for f,_,_,_ in p)==900
    assert sum(f=="validation" for f,_,_,_ in p)==240
    assert set(y for f,y,_,_ in p if f=="train")==set(range(1980,2005))
    assert set(y for f,y,_,_ in p if f=="validation")==set(range(2005,2015))


def test_no_future_train_injection():
    p=schedule()
    p[0]=("train",2030,1,6)
    with pytest.raises(ValueError,match="train year"):
        validate_schedule(p)


def test_native_noleap_february_never_creates_feb29():
    import cftime
    units="days since 1980-01-01 12:00:00"
    src=cftime.datetime(1980,2,26,12,calendar="noleap")
    dst=cftime.datetime(1980,2,27,12,calendar="noleap")
    arr=np.array(cftime.date2num([src,dst],units,calendar="noleap"))
    i,j,a,b=native_noleap_pair(arr,1980,2,26,units,"noleap")
    assert (i,j)==(0,1)
    assert a.startswith("1980-02-26") and b.startswith("1980-02-27")


def test_training_p95_fits_75_per_month_only():
    months=np.repeat(np.arange(1,13),75)
    years=np.full(len(months),1980,dtype=int)
    train=np.ones(len(months),dtype=bool)
    y=np.ones((len(months),1,2,3),dtype=np.float32)*285
    y[0]=300
    q,c=monthly_training_reference(y,years,months,train)
    assert q.shape==(12,2,3)
    assert set(c.values())=={75}
    train[0]=False
    with pytest.raises(ValueError,match="exactly 75"):
        monthly_training_reference(y,years,months,train)
