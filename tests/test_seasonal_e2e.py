"""End-to-end seasonal threshold + latitude-weighting contract, no cloud access."""
import numpy as np
from climate_shift.compare_e1 import compare


def test_full_seasonal_weighted_comparison_without_test_leakage(tmp_path):
    rng = np.random.default_rng(2026)
    dates = np.array(
        [np.datetime64(f"{year}-{month:02d}-{day:02d}T00", "ns")
         for year in (2008, 2009, 2010)
         for month in range(1, 13)
         for day in range(2, 22)]
    )
    assert len(dates) == 720
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    month_indices = dates.astype("datetime64[M]").astype(int) % 12 + 1
    seasonal = (8 * np.sin(2 * np.pi * (month_indices - 1) / 12)).reshape(-1, 1, 1, 1)
    latitude = np.linspace(25, 50, 8).astype("float32")
    x = (280 + seasonal + rng.normal(0, 3, size=(720, 1, 8, 8))).astype("float32")
    y = (x + rng.normal(0, .8, size=x.shape)).astype("float32")
    path = tmp_path / "seasonal.npz"
    np.savez_compressed(path, X=x, y=y, years=years,
                        target_times=dates, source_times=dates - np.timedelta64(24, "h"),
                        latitude=latitude)

    report = compare(path, tmp_path / "metrics.json", 2008, 2009, 2010,
                     epochs=1, batch_size=16, width=4, modes=2,
                     require_pair_times=True)
    assert report["split_isolation"]["status"] == "passed"
    assert report["area_weighting_status"] == "cos(latitude)"
    assert report["seasonal_local_p95_definition"]["status"] == "available_training_only"
    assert set(report["seasonal_local_hot_diagnostics"]) == {"persistence", "fno", "unet"}
    assert set(report["latitude_weighted_metrics"]) == {"persistence", "fno", "unet"}
    assert report["seasonal_local_p95_definition"]["actual_samples_per_month"]["01"] == 20
    assert report["n_samples"] == {"train": 240, "validation": 240, "test": 240}
    for model in ("persistence", "fno", "unet"):
        assert np.isfinite(report["latitude_weighted_metrics"][model]["rmse"])
        assert report["seasonal_local_hot_diagnostics"][model]["weighting"] == "cos(latitude)"
