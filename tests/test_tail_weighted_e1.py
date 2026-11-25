"""Controlled local-month tail-aware loss: validate historical train-only conditioning."""
import numpy as np
import pytest
from climate_shift.compare_e1 import compare


def _make_monthly_data(tmp_path):
    rng = np.random.default_rng(91)
    times = np.array([
        np.datetime64(f"{yr}-{mo:02d}-{day:02d}", "ns")
        for yr in (2008, 2009, 2010, 2011, 2012)
        for mo in range(1, 13)
        for day in range(1, 21)
    ])
    years = times.astype("datetime64[Y]").astype(int) + 1970
    month = (times.astype("datetime64[M]").astype(int) % 12).astype(float)
    y = (281 + 5 * np.sin(month[:,None,None,None]/12*np.pi*2)
         + rng.normal(0, 3, size=(len(times),1,8,8))).astype("float32")
    x = y + rng.normal(0, 1, size=y.shape).astype("float32")
    path = tmp_path/"dataset.npz"
    np.savez_compressed(path, X=x,y=y,years=years,
                        target_times=times,
                        latitude=np.linspace(25,50,8))
    return path


def test_weighted_training_logs_tail_loss_and_tail_metrics(tmp_path):
    path = _make_monthly_data(tmp_path)
    report = compare(path, tmp_path/"tail.json", 2010, 2011, 2012,
                     epochs=1, width=4, modes=2, tail_weight=3.0, seed=11)
    assert report["hyperparameters"]["tail_weight"] == 3.0
    assert report["seasonal_local_p95_definition"]["status"] == "available_training_only"
    assert report["conditional_skill_reference"] == "train_only_month_local_mean_std"
    assert len(report["diagnostics"]["fno"]["history"]) == 1
    assert report["seasonal_local_hot_diagnostics"]["fno"]["hot_recall"] is not None


def test_tail_training_requires_training_only_monthly_reference(tmp_path):
    rng=np.random.default_rng(4)
    x=rng.normal(size=(12,1,8,8)).astype("float32")
    p=tmp_path/"legacy.npz"
    np.savez_compressed(p, X=x, y=x+0.4, years=np.repeat([2008,2011,2012],4))
    with pytest.raises(ValueError,match="requires complete training-only"):
        compare(p, tmp_path/"fail.json", 2008, 2011, 2012,
                width=4, modes=2, epochs=1, tail_weight=2.0)


def test_tail_weight_reject_negative(tmp_path):
    with pytest.raises(ValueError,match="Invalid training hyperparameters"):
        compare(tmp_path/"missing.npz",tmp_path/"bad.json",2008,2011,2012,
                tail_weight=-1.0)
