import numpy as np
import pytest
from climate_shift.historical_shift import audit_historical_shift


def create_sample(path, warm_test=False):
    rng = np.random.default_rng(4)
    dates = np.asarray([
        np.datetime64(f"{year}-{month:02d}-{day:02d}", "ns")
        for year in (2008, 2009, 2010, 2012, 2013, 2014)
        for month in range(1, 13)
        for day in range(1, 25)
    ])
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    n = len(years)
    warming = (years >= 2012).astype(float) * (3.0 if warm_test else 0.0)
    seasonal = np.cos(months * np.pi / 6) * 5
    lat = np.array([25.0, 45.0])
    temp = (280 + seasonal[:, None, None] + warming[:, None, None] +
            rng.normal(0, 1, (n, 2, 2))).astype("float32")
    y = temp[:, None]
    np.savez_compressed(path, X=y, y=y, years=years, target_times=dates,
                        latitude=lat)
    return path


def test_training_only_reference_and_warm_heldout(tmp_path):
    baseline = create_sample(tmp_path / "warm.npz", warm_test=True)
    result = audit_historical_shift(baseline)
    assert result["train_only_month_local_climatology"] is True
    assert max(abs(result["per_year"][str(yr)]["mean_anomaly_K"]) for yr in (2008,2009,2010)) < 1.0
    assert result["per_year"]["2012"]["mean_anomaly_K"] > 2
    assert result["per_year"]["2013"]["mean_standardized_anomaly"] > 1
    assert result["train_counts_by_month"]["01"] == 72
    assert result["per_year"]["2014"]["n_pairs"] == 288
    assert "climate_change_attribution" in result["status"]


def test_missing_training_month_fails_closed(tmp_path):
    p = create_sample(tmp_path / "few.npz")
    with np.load(p) as d:
        keep = ~((d["years"] <= 2010) & (
            d["target_times"].astype("datetime64[M]").astype(int) % 12 + 1 == 1))
        arrays = {name: d[name][keep] if name in ("X","y","years","target_times") else d[name]
                  for name in d.files}
    np.savez_compressed(p, **arrays)
    with pytest.raises(ValueError, match="fewer than 20"):
        audit_historical_shift(p)
