"""No-SSP CMIP6 Historical neural fitting safety checks."""
import pytest
from scripts.fit_cmip6_awi_historical import verify_historical_only, fit_historical


def test_missing_or_unfrozen_historical_data_rejected(tmp_path):
    p=tmp_path/"made-up.npz"
    p.write_bytes(b"not the audited source")
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        verify_historical_only(p)


def test_training_configuration_is_locked_before_reading_data(tmp_path):
    with pytest.raises(ValueError,match="unfrozen"):
        fit_historical("not-here.npz",tmp_path/"o",epochs=101)
    with pytest.raises(ValueError,match="unfrozen"):
        fit_historical("not-here.npz",tmp_path/"o",tail_weight=1.5)
    with pytest.raises(ValueError,match="unfrozen"):
        fit_historical("not-here.npz",tmp_path/"o",width=16)
