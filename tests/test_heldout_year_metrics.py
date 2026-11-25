import numpy as np
from climate_shift.compare_e1 import compare


def test_scores_each_heldout_year_without_refit(tmp_path):
    rng = np.random.default_rng(88)
    # 2008 training, 2011 validation, 2012-14 three independent held-out years
    years = np.repeat([2008, 2011, 2012, 2013, 2014], 3)
    x = (280 + rng.normal(0, 2, (len(years), 1, 8, 8))).astype("float32")
    y = x + rng.normal(0, .4, x.shape).astype("float32")
    path = tmp_path / "cohort.npz"
    np.savez_compressed(path, X=x, y=y, years=years, latitude=np.linspace(25, 50, 8))
    result = compare(path, tmp_path / "results.json", 2008, 2011, 2014,
                     epochs=1, width=4, modes=2, seed=3)
    assert result["n_samples"] == {"train": 3, "validation": 3, "test": 9}
    assert set(result["per_test_year_metrics"]) == {"persistence", "fno", "unet"}
    for model in ("persistence", "fno", "unet"):
        assert set(result["per_test_year_metrics"][model]) == {"2012", "2013", "2014"}
        for yr in ("2012", "2013", "2014"):
            row = result["per_test_year_metrics"][model][yr]
            assert row["n_samples"] == 3
            assert np.isfinite(row["area_weighted"]["rmse"])
            assert row["monthly_local_hot"] is None
    assert result["seasonal_local_p95_definition"]["status"].startswith("unavailable")
