"""Year-by-year aggregate tests: do not pool distinct meteorological years."""
import numpy as np


def test_yearly_multiseed_aggregate(tmp_path):
    rng = np.random.default_rng(31)
    years = np.repeat([2008, 2011, 2012, 2013, 2014], 4)
    x = (280 + rng.normal(0, 2, (len(years), 1, 8, 8))).astype("float32")
    y = (x + rng.normal(0, .3, x.shape)).astype("float32")
    source = tmp_path / "data.npz"
    np.savez_compressed(source, X=x, y=y, years=years,
                        latitude=np.linspace(25, 50, 8))
    from scripts.run_e1_multiseed import run_seeds
    summary = run_seeds(str(source), str(tmp_path / "output"), 2008, 2011, 2014,
                        seeds=(11, 22), epochs=1, width=4, modes=2)
    assert summary["heldout_years"] == ["2012", "2013", "2014"]
    for year in summary["heldout_years"]:
        for model in ("persistence", "fno", "unet"):
            row = summary["per_test_year_models"][year][model]
            assert row["n_samples"] == 4
            metric = row["area_weighted_rmse"]
            assert len(metric["individual"]) == 2
            assert metric["mean"] == np.mean(metric["individual"])
            assert row["monthly_local_hot_recall"]["mean"] is None
