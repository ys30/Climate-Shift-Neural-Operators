"""E1 v5.11 independence of calendar schedule and source-grid guards."""
import copy
import pytest
from scripts.audit_cmip6_v511_new_2040_cohort import schedule,validate_schedule,validate_records,DAYS
def example():
    records=[]
    for g in ("AWI","BCC"):
        for s in ("ssp245","ssp585"):
            records.append(dict(gcm=g,scenario=s,n_pairs=84,
                latitude_sha256=g+"lat",longitude_sha256=g+"lon",
                grid_shape=[26,27] if g=="AWI" else [23,23],
                daily_records=[{"month":m,"source_day":d}
                               for m in range(1,13) for d in DAYS]))
    return records

def test_frozen_new_schedule():
    s=schedule()
    validate_schedule(s)
    assert len(s)==84
    assert not any(d in (8,20) for _,_,d in s)

def test_changed_day_rejected():
    s=list(schedule())
    s[0]=(2040,1,8)
    with pytest.raises(ValueError,match="Non-frozen"):validate_schedule(s)

def test_complete_native_cohorts():
    assert validate_records(example())

def test_cross_ssp_grid_drift_rejected():
    s=example()
    s[1]["latitude_sha256"]="bad"
    with pytest.raises(ValueError,match="grid mismatch"):validate_records(s)

def test_missing_pair_rejected():
    s=example()
    s[2]["daily_records"].pop()
    with pytest.raises(ValueError,match="sequence"):validate_records(s)

def test_missing_scenario_rejected():
    with pytest.raises(ValueError,match="Missing"):validate_records(example()[:-1])
