"""v5.12 safety gates: read-only archived 2040 pilot and complete frozen models."""
import copy
import pytest
from scripts.evaluate_cmip6_v512_new_2040_frozen import verify_pilot,score_model
from scripts.audit_cmip6_v511_new_2040_cohort import DAYS
import json

def sample():
    rows=[]
    for g in ("AWI","BCC"):
        for s in ("ssp245","ssp585"):
            rows.append({
                "gcm":g,"scenario":s,"n_pairs":84,
                "grid_shape":[26,27] if g=="AWI" else [23,23],
                "latitude_sha256":g+"lat","longitude_sha256":g+"lon",
                "daily_records":[{"month":m,"source_day":d} for m in range(1,13) for d in DAYS]})
    return {"status":"v511_verified_new_2040_seven_dates_per_month_source_cohort",
            "source_only":True,"year":2040,"days_per_month":list(DAYS),"rows":rows}

def test_exact_pilot_gate(tmp_path):
    p=tmp_path/"input.json";p.write_text(json.dumps(sample()))
    out=verify_pilot(p)
    assert len(out)==4

def test_nonfrozen_day_fails(tmp_path):
    d=sample();d["days_per_month"]=[8,20]
    p=tmp_path/"input.json";p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="Unverified"):verify_pilot(p)

def test_changed_grid_fails(tmp_path):
    d=sample();d["rows"][1]["latitude_sha256"]="different"
    p=tmp_path/"input.json";p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="grid mismatch"):verify_pilot(p)

def test_missing_2040_day_fails(tmp_path):
    d=sample();d["rows"][1]["daily_records"].pop()
    p=tmp_path/"input.json";p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="sequence"):verify_pilot(p)
